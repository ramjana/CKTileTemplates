// input:
//   A
//   B0
//   D00, D01, ...
//   B1
//   D10, D11, ...
// output:
//   E1
// C0 = A * B0
// E0 = elementwise_op(C0, D00, D01, ...)
// F0 = softmax(E0)
// C1 = F0 * B1
// E1 = elementwise_op(C1, D10, D11, ...)c

struct BatchedGemmSoftmaxGemm
{
    __host__ __device__ void
    operator()(ProgramServer& ps,
               const std::array<index_t, 3> a_b_m_k_lengths,
               const std::array<index_t, 3> a_b_m_k_strides,
               const std::array<index_t, 3> b0_b_n_k_lengths,
               const std::array<index_t, 3> b0_b_n_k_strides,
               const std::array<const std::array<index_t, 3>, NumDTensor> d0s_b_m_n_lengths,
               const std::array<const std::array<index_t, 3>, NumDTensor> d0s_b_m_n_strides,
               const std::array<index_t, 3> b1_b_n_l_lengths,
               const std::array<index_t, 3> b1_b_n_l_strides,
               const std::array<const std::array<index_t, 3>, NumDTensor> d1s_b_m_l_lengths,
               const std::array<const std::array<index_t, 3>, NumDTensor> d1s_b_m_l_strides,
               const std::array<index_t, 3> e_b_m_l_lengths,
               const std::array<index_t, 3> e_b_m_l_strides,
               //
               const T* p_a,
               const T* p_b0,
               const std::array<const T*> p_d0s,
               const T* p_b1,
               const std::array<const T*> p_d1s,
               T* p_e)
    {
        const auto a  = ps(make_naive_tensor(a_b_m_k_lengths, a_b_m_k_strides), p_a);
        const auto b0 = ps(make_naive_tensor(b0_b_n_k_lengths, b0_b_n_k_strides), p_b0);
        const auto b1 = ps(make_naive_tensor(b1_b_n_l_lengths, b1_b_n_l_strides), p_b1);

        const auto d0s = ps(generate_tuple(
            [&](auto i) {
                return make_naive_tensor(d0s_b_m_n_lengths[i], d0s_b_m_n_strides[i], p_d0s[i]),
            },
            Number<NumDTensor>{}));

        const auto d1s = ps(generate_tuple(
            [&](auto i) {
                return make_naive_tensor(d1s_b_m_l_lengths[i], d1s_b_m_l_strides[i], p_d1s[i]),
            },
            Number<NumDTensor>{}));

        auto e = ps(make_naive_tensor(e_m_l_lengths, e_m_l_strides), p_e);

        // divide problem
        const auto num_b = e_b_m_l_lengths[0];
        const auto num_m = e_b_m_l_lengths[1];
        const auto num_l = e_b_m_l_lengths[2];

        const auto num_tile_b = num_b;
        const auto num_tile_m = num_m / kMPerTile;
        const auto num_tile_l = num_l / kLPerTile;

        const auto block2tile =
            ps(make_cluster_descriptor(make_tuple(num_tile_b, num_tile_m, num_tile_l)));

        const auto idx_tile = block2tile.CalculateBottonIndex(make_tuple(get_block_1d_id();));

        const auto idx_tile_b = idx_tile[I0];
        const auto idx_tile_m = idx_tile[I1];
        const auto idx_tile_n = idx_tile[I2];

        // A/B copy
        auto window_a_dram = make_window(a_dram_global,
                                         {kMPerTile, kKPerTile},
                                         {id_tile_m * kMPerTile, 0},
                                         a_dram_window_map_strategy);

        auto window_b0_dram = make_window(
            b0_dram_global, {kNPerTile, kKPerTile}, {0, 0}, b0_dram_window_map_strategy);

        // d should same distribution as c for best performance
        auto window_d00_dram = make_window(d00_dram_global,
                                           {kMPerTile, kNPerTile},
                                           {id_tile_m * kMPerTile, 0},
                                           d_dram_window_strategy);

        for(index_t iN = 0; iN <= numN; iN += kNPerBlock)
        {
            // Gemm0 multiple-D
            for(index_t iK = 0; iK < numK; iK += kKPerBlock)
            {
                auto a_vgpr_block  = load(window_a_dram, a_dram_load_strategy);
                auto b0_vgpr_block = load(window_b0_dram, b0_dram_load_strategy);

                store(a_vgpr_block, a_lds_block, a_lds_store_strategy);
                store(b0_vgpr_block, b0_lds_block, b0_lds_store_strategy);

                block_sync_lds();

                block_gemm.dot_product_accumulate(c0_vgpr_block, a_lds_block, b0_lds_block);

                block_sync_lds();

                window_a_dram += {0, 0, kKPerBlock};
                window_b0_dram += {0, 0, kKPerBlock};
            }

            // e should same distribution as c for best performance
            auto e0_vgpr_block =
                make_distributed_tensor({kMPerTile, kNPerTile}, e_vgpr_block_strategy);

            // CDE pointwise
            elementwise_op(c0_vgpr_block, d00_vgpr_block, e0_vgpr_block, cde_element_op);

            // local softmax, this is a distributed tensor
            auto e0_max_1d = reduce(e0_vgpre_block, reduce_dims_xxxx, max{});

            auto e0_max_2d = transform_tensor(e0_max_1d, make_broadcast_transform(xxx));

            auto e0_sum_1d =
                reduce(e0_vgpre_block, e0_max_2d, reduce_dims_xxxx, [&](auto x, auto y) {
                    return x > y ? x : y;
                });

            auto e0_sum_2d = transform_tensor(e0_sum_1d, make_broadcast_transform(xxx));

            // f0 is the A matrix for 2nd GEMM
            auto f0 = position_aware_elementwise_op(
                e0_vgpr_block, e0_max_2d, e0_sum_2d, [&](auto idx, auto x, auto max, auto sum) {
                    auto m = idx[0] + id_tile_m * kMPerTile;
                    auto n = idx[1] + iN;

                    return m > n ? 0 : math::exp(x - max) / sum;
                });

            // 2nd GEMM
            auto b1_vgpr_block = load(window_b1_dram, b1_dram_load_strategy);

            store(b1_vgpr_block, b1_lds_block, b1_lds_store_strategy);

            block_sync_lds();

            block_gemm1.dot_product_accumulate(c1_vgpr_block, f0_vgpr_block, b1_lds_block);

            block_sync_lds();

            window_b1_dram += {0, kNPerBlock, 0};
        }

        // CDE pointwise
        elementwise_op(c1_vgpr_block, d10_vgpr_block, e1_vgpr_block, cde1_element_op);
    }
};
