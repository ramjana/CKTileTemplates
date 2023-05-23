// C = A * B
// E = func(C, D0, D1, ...);
template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct GemmMultiD
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    __host__ __device__ void
    operator()(ProgramServer& ps,
               const std::array<index_t, 2> a_m_k_lengths,
               const std::array<index_t, 2> a_m_k_strides,
               const std::array<index_t, 2> b_n_k_lengths,
               const std::array<index_t, 2> b_n_k_strides,
               const std::array<const std::array<index_t, 2>, NumDTensor> ds_m_n_lengths,
               const std::array<const std::array<index_t, 2>, NumDTensor> ds_m_n_strides,
               const std::array<index_t, 2> e_m_n_lengths,
               const std::array<index_t, 2> e_m_n_strides,
               //
               const T* p_a,
               const T* p_b,
               const std::array<const T*> p_ds,
               T* p_e)
    {
        using namespace ck;

        const auto a  = ps(make_naive_tensor(a_m_k_lengths, a_m_k_strides), p_a);
        const auto b  = ps(make_naive_tensor(b_n_k_lengths, b_n_k_strides), p_b);
        const auto ds = ps(generate_tuple(
            [&](auto i) {
                return make_naive_tensor(ds_m_n_lengths[i], ds_m_n_strides[i], p_ds[i]),
            },
            Number<NumDTensor>{}));
        auto e        = ps(make_naive_tensor(e_m_n_lengths, e_m_n_strides), p_e);

        // divide problem
        const auto num_m = e_m_n_lengths[0];
        const auto num_n = e_m_n_lengths[1];

        const auto id_block = get_block_1d_id();

        const auto num_tile_m = num_gemmm / MPerTile;
        const auto num_tile_n = num_gemmn / NPerTile;

        const auto block2tile = ps(make_cluster_descriptor(make_tuple(num_tile_m, num_tile_n)));

        const auto id_tile = block2tile.CalculateBottonIndex(id_block);

        const auto id_tile_m = id_tile.At<0>();
        const auto id_tile_n = id_tile.At<1>();

        // A/B in DRAM
        // A/B DRAM layout is part of problem, not solution
        // DO NOT let user know there is optimization on tensor transform on A/B DRAM tensor
        const auto a_dram_global = ps(make_naive_tensor(a_m_k_lengths, a_m_k_strides), p_a_dram);
        const auto b_dram_global = ps(make_naive_tensor(b_n_k_lengths, b_n_k_strides), p_b_dram);

        // A/B tile in LDS
        // A/B DRAM layout is part of solution
        ADataType* p_a_lds = shared_memmory.get_pointer(0);

        // [allow optimization] allow different LDS layouts
        constexpr auto a_lds_block =
            make_tensor(p_a_lds, {kMPerBlock, kKPerBlock}, a_lds_block_strategy);

        constexpr auto a_lds_byte = a_lds_block.get_num_of_byte();

        BDataType* p_b_lds = shared_memory.get_aligned_pointer(a_lds_byte);

        // [allow optimization] allow different LDS layouts
        constexpr auto b_lds_block =
            make_tensor({p_b_lds, kNPerBlock, kKPerBlock}, b_lds_block_strategy);

        // A/B copy
        auto window_a_dram = make_window(a_dram_global,
                                         {MPerTile, KPerTile},
                                         {id_tile_m * MPerTile, id_tile_k * KPerTile},
                                         a_dram_window_map_strategy);

        auto window_a_block =
            make_window(a_lds_block, {MPerTile, KPerTile}, {0, 0}, a_lds_window_map_strategy);

        // block GEMM
        // operation-based syntax: per-operation solution strategy
        auto block_gemm = make_block_gemm(a_lds_block, b_lds_block, block_gemm_strategy);

        // Distributed C in VGPR
        auto c_vgpr_block = decltype(block_gemm.dot_product(a_lds_block, b_lds_block)){};

        for(index_t k = 0; k < K; k += kKPerBlock)
        {
            auto a_vgpr_block_tmp = load(window_a_dram, a_dram_load_strategy);
            auto b_vgpr_block_tmp = load(window_b_dram, b_dram_load_strategy);

            auto a_vpgr_block = elementwise_op(a_vgpr_block_tmp, a_element_op);
            auto b_vpgr_block = elementwise_op(b_vgpr_block_tmp, b_element_op);

            store(a_vgpr_block, a_lds_block, a_lds_store_strategy);
            store(b_vgpr_block, b_lds_block, b_lds_store_strategy);

            block_sync_lds();

            block_gemm.dot_product_accumulate(c_vgpr_block, a_lds_block, b_lds_block);

            block_sync_lds();

            window_a_dram += {0, kKPerBlock};
            window_b_dram += {0, kKPerBlock};
        }

        // shuffle C
        auto p_lds_cshuffle_workspace = xxx;

        auto c_vgpr_block_shuffled =
            make_distributed_tensor({kKMerBlock, kNPerBlock}, c_vgpr_block_shuffled_strategy);

        shuffle_distributed_tensor(c_vgpr_block, v_vgpr_block_shuffled, p_lds_cshuffle_workspace);

        // pointwise
        auto window_d_dram = make_window(d_dram_global,
                                         {kMPerTile, kNPerTile},
                                         {id_tile_m * kMPerTile, id_tile_n * kNPerTile},
                                         d_dram_window_strategy);

        auto window_e_dram = make_window(e_dram_global,
                                         {kMPerTile, kNPerTile},
                                         {id_tile_m * kMPerTile, id_tile_n * kNPerTile},
                                         e_dram_window_strategy);

        auto e_vgpr_block_slice =
            elementwise_op(c_vgpr_block_slice, window_d_dram, cd_elementwise_op);

        // store E
        store(e_vgpr_block_slice, window_e_dram);
    }
};
