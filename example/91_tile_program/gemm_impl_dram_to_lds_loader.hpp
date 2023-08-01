template <typename ADataType,
          typename BDataType,
          ck::index_t kBlockSize,
          ck::index_t kMPerBlock,
          ck::index_t kNPerBlock,
          ck::index_t kKPerBlock>
struct NaiveDram2LdsLoader
{
    __host__ __device__ static constexpr auto MakeADramTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<4, 64>, Sequence<1, 4, 8>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0>>,
                                           Sequence<2, 2>,
                                           Sequence<1, 2>>{});
    }

    __host__ __device__ static constexpr auto MakeBDramTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<4, 32>, Sequence<2, 2, 8>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0>>,
                                           Sequence<2, 2>,
                                           Sequence<1, 2>>{});
    }
};

template <typename ADataType,
          typename BDataType,
          ck::index_t kBlockSize,
          ck::index_t kMPerBlock,
          ck::index_t kNPerBlock,
          ck::index_t kKPerBlock>
struct BetterDram2LdsLoader
{
    __host__ __device__ static constexpr auto MakeADramTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        constexpr index_t kK1 = 16 / sizeof(ADataType);
        constexpr index_t kK0 = kKPerBlock / kK1;
        constexpr index_t kM2 = get_warp_size() / kK0;
        constexpr index_t kM1 = kBlockSize / get_warp_size();
        constexpr index_t kM0 = kMPerBlock / (kM2 * kM1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<kM0, kM1, kM2>, Sequence<kK0, kK1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
    }

    __host__ __device__ static constexpr auto MakeBDramTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        constexpr index_t kK1 = 16 / sizeof(BDataType);
        constexpr index_t kK0 = kKPerBlock / kK1;
        constexpr index_t kN2 = get_warp_size() / kK0;
        constexpr index_t kN1 = kBlockSize / get_warp_size();
        constexpr index_t kN0 = kNPerBlock / (kN2 * kN1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<kN0, kN1, kN2>, Sequence<kK0, kK1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
    }
};
