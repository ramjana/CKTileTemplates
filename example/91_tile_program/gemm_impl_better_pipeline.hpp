// C = A * B
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementWiseOperation,
          typename BElementWiseOperation,
          typename CElementWiseOperation,
          ck::index_t kBlockSize,
          ck::index_t kMPerBlock,
          ck::index_t kNPerBlock,
          ck::index_t kKPerBlock,
          typename LdsAllocator,
          typename Dram2LdsLoader>
struct GemmBetterPipeline
{
    __host__ __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        return ck::math::integer_divide_ceil(
                   sizeof(ADataType) *
                       LdsAllocator::MakeALdsBlockDescriptor().GetElementSpaceSize(),
                   16) *
                   16 +
               sizeof(BDataType) * LdsAllocator::MakeBLdsBlockDescriptor().GetElementSpaceSize();
    }

    __host__ __device__ void operator()(ProgramServer& ps,
                                        const ADataType* p_a,
                                        const BDataType* p_b,
                                        CDataType* p_c,
                                        ck::index_t M,
                                        ck::index_t N,
                                        ck::index_t K,
                                        ck::index_t Lda,
                                        ck::index_t Ldb,
                                        ck::index_t Ldc,
                                        AElementWiseOperation /* a_op */,
                                        BElementWiseOperation /* b_op */,
                                        CElementWiseOperation /* c_op */)
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        __shared__ char p_shared_char[GetStaticLdsSize()];

        // FIXME: assume RCR layout
        const auto a_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a, make_tuple(M, K), make_tuple(Lda, 1), Number<32>{}, Number<1>{});

        const auto b_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b, make_tuple(N, K), make_tuple(Ldb, 1), Number<32>{}, Number<1>{});

        // divide problem
        const auto id_block = ps.get_block_1d_id();

        const auto num_tile_m = M / kMPerBlock;
        const auto num_tile_n = N / kNPerBlock;

        const auto block2tile = ps(make_cluster_descriptor(make_tuple(num_tile_m, num_tile_n)));

        const auto id_tile = block2tile.CalculateBottomIndex(make_tuple(id_block));

        const auto iM = ps.read_first_lane(id_tile.At<0>() * kMPerBlock);
        const auto iN = ps.read_first_lane(id_tile.At<1>() * kNPerBlock);

        // A tile in LDS
        ADataType* p_a_lds = static_cast<ADataType*>(static_cast<void*>(p_shared_char));

        // [allow optimization] allow different LDS layouts
        constexpr auto a_lds_block_desc = LdsAllocator::MakeALdsBlockDescriptor();

        auto a_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_a_lds, a_lds_block_desc);

        constexpr index_t a_lds_block_space_size_aligned =
            math::integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.GetElementSpaceSize(),
                                      16) *
            16;

        // B tile in LDS
        BDataType* p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(p_shared_char + a_lds_block_space_size_aligned));

        // [allow optimization] allow different LDS layouts
        constexpr auto b_lds_block_desc = LdsAllocator::MakeBLdsBlockDescriptor();

        auto b_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_b_lds, b_lds_block_desc);

        // A DRAM tile window
        auto a_copy_dram_window =
            make_tile_window(a_dram_grid,
                             make_tuple(Number<kMPerBlock>{}, Number<kKPerBlock>{}),
                             {iM, 0},
                             Dram2LdsLoader::MakeADramTileDistribution());

        auto a_copy_lds_window =
            make_tile_window(a_lds_block,
                             make_tuple(Number<kMPerBlock>{}, Number<kKPerBlock>{}),
                             {0, 0},
                             a_copy_dram_window.GetTileDistribution());

        // B DRAM tile window
        auto b_copy_dram_window =
            make_tile_window(b_dram_grid,
                             make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}),
                             {iN, 0},
                             Dram2LdsLoader::MakeBDramTileDistribution());

        auto b_copy_lds_window =
            make_tile_window(b_lds_block,
                             make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}),
                             {0, 0},
                             b_copy_dram_window.GetTileDistribution());

        // A tile for block GEMM
        auto a_lds_gemm_window = make_tile_window(
            a_lds_block, make_tuple(Number<kMPerBlock>{}, Number<kKPerBlock>{}), {0, 0});

        // A tile for block GEMM
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block, make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}), {0, 0});

        // Acc tile
        auto acc_block_tile = decltype(block_gemm_cr_as_bs(a_lds_gemm_window, b_lds_gemm_window)){};

        // prefetch
        // global read 0
        auto a_block_tile = load_tile(a_copy_dram_window);
        auto b_block_tile = load_tile(b_copy_dram_window);

        // move to 1
        move_tile_window(a_copy_dram_window, {0, kKPerBlock});
        move_tile_window(b_copy_dram_window, {0, kKPerBlock});

        // Initialize C
        tile_elementwise_inout([](auto& acc) { acc = 0; }, acc_block_tile);

        // LDS write 0
        store_tile(a_copy_lds_window, a_block_tile);
        // global read 1
        a_block_tile = load_tile(a_copy_dram_window);

        // LDS write 0
        store_tile(b_copy_lds_window, b_block_tile);
        // global read 1
        b_block_tile = load_tile(b_copy_dram_window);

        index_t iK = 0;

        do
        {
            ps.block_sync_lds();

            // GEMM i
            block_gemm_cr_as_bs(acc_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            ps.block_sync_lds();

            // move to i + 2
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            // LDS write i + 1
            store_tile(a_copy_lds_window, a_block_tile);
            // global read i + 2
            a_block_tile = load_tile(a_copy_dram_window);

            // LDS write i + 1
            store_tile(b_copy_lds_window, b_block_tile);
            // global read i + 2
            b_block_tile = load_tile(b_copy_dram_window);

            iK += kKPerBlock;

        } while(iK < K - 2 * kKPerBlock);

        // tail
        {
            ps.block_sync_lds();

            // GEMM num_loop - 2
            block_gemm_cr_as_bs(acc_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            ps.block_sync_lds();

            // LDS write num_loop - 1
            store_tile(a_copy_lds_window, a_block_tile);
            store_tile(b_copy_lds_window, b_block_tile);

            ps.block_sync_lds();

            // GEMM num_loop - 1
            block_gemm_cr_as_bs(acc_block_tile, a_lds_gemm_window, b_lds_gemm_window);
        }

        // type convert
        auto c_block_tile = tile_elementwise_in(
            [](const auto& acc) { return type_convert<CDataType>(acc); }, acc_block_tile);

        // store C
        auto c_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_c, make_tuple(M, N), make_tuple(Ldc, 1), Number<32>{}, Number<1>{});

        auto c_dram_window =
            make_tile_window(c_dram_grid,
                             make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}),
                             {iM, iN},
                             c_block_tile.GetTileDistribution());

        store_tile(c_dram_window, c_block_tile);
    }
};
