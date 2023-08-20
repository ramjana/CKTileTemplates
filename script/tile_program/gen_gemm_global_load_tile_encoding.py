import sys
import os
import itertools
import numpy as np
import enum

# just generate to current path is OK
OUTPUT_FILE='gemm_global_load_tile_encoding_predef.hpp'

B_ARRAY=[256, 128, 64]          # block size
S_ARRAY=[32, 64, 96, 128, 160, 192, 256]  # M/N dim 
R_ARRAY=[8, 16, 32, 64, 128]     # K dim, reduce dim
# V_ARRAY=[8, 4, 2, 1]        # vector size, alignment

WAVE_SIZE = 64

class GemmLayout(enum.Enum):
    S_R = enum.auto()
    R_S = enum.auto()

M_ARRAY=[GemmLayout.S_R, GemmLayout.R_S]

class naive_tensor_desc(object):
    # dense nd desc
    def __init__(self, length : list, stride = None):
        self.length = length
        def exclusive_scan_(x : list):
            a = [1]
            for i in range(len(x) - 1, 0, -1):  # exclusive the first one
                t = a[0] * x[i]
                a.insert(0, t)
            return a
        self.stride = stride if stride != None else exclusive_scan_(length)
        def reduce_mul_(x : list):
            r = 1
            for i in x:
                r = r * i
            return r
        self.total_length = reduce_mul_(length)

    def offset(self, coord : list):
        assert len(coord) == len(self.stride)
        acc = 0
        for i in range(len(coord)):
            assert coord[i] < self.length[i], f'i:{i}, {coord[i] }, {self.length[i]}'
            acc = acc + coord[i] * self.stride[i]
        return acc

    def dump(self):
        print(f'length:{self.length}')
        print(f'stride:{self.stride}')

class gemm_thread_group_desc(object):
    def __init__(self, size, t_len, c_len, layout, wave_size = WAVE_SIZE):
        self.size = size
        self.t_len = t_len
        self.c_len = c_len
        self.layout = layout    # TODO: this field may or may not be used as template name
        self.wave_size = wave_size

        assert(len(t_len) == 4 and len(c_len))

        self.desc_x = naive_tensor_desc([c_len[0], t_len[0], c_len[1], t_len[1]])
        self.desc_y = naive_tensor_desc([c_len[2], t_len[2], c_len[3], t_len[3]])

        self.dim_x = self.desc_x.total_length
        self.dim_y = self.desc_y.total_length

    def to_enc_str(self):
        '''
            template <typename RsLengths_,    // Sequence<...>
                    typename HsLengthss_,   // Tuple<Sequence<...>, ...>
                    typename Ps2RHssMajor_, // Tuple<Sequence<...>, ...>
                    typename Ps2RHssMinor_, // Tuple<Sequence<...>, ...>
                    typename Ys2RHsMajor_,  // Sequence<...>
                    typename Ys2RHsMinor_   // Sequence<...>
                    >
            struct StaticTensorDistributionEncoding
        '''
        ws = self.wave_size
        tx_0,   _ , ty_0, ty_1 = self.t_len[0], self.t_len[1], self.t_len[2], self.t_len[3]
        _   , cx_1,   _ , cy_1 = self.c_len[0], self.c_len[1], self.c_len[2], self.c_len[3]

        if cy_1 > ws:
            assert cy_1 % ws == 0, f't:{self.t_len}, c:{self.c_len}, ws:{ws}' # TODO: Maybe no need even divide?
            if ty_0 != 1:
                dim_x, dim_y = [tx_0, cx_1], [ty_0, cy_1 // ws, ws, ty_1]
                p_0_i, p_0_j = [1, 2], [1, 1]
                p_1_i, p_1_j = [2], [2]
                y_i, y_j = [1, 2, 2], [0, 0, 3]
            else:
                dim_x, dim_y = [tx_0, cx_1], [cy_1 // ws, ws, ty_1]
                p_0_i, p_0_j = [1, 2], [1, 0]
                p_1_i, p_1_j = [2], [1]
                y_i, y_j = [1, 2], [0, 2]
        else:
            r = ws // cy_1
            assert cx_1 % r == 0, f't:{self.t_len}, c:{self.c_len}, r:{r}, ws:{ws}'
            if ty_0 != 1:
                dim_x, dim_y = [tx_0, cx_1 // r, r], [ty_0, cy_1, ty_1]
                p_0_i, p_0_j = [1], [1]
                p_1_i, p_1_j = [1, 2], [2, 1]
                y_i, y_j = [1, 2, 2], [0, 0, 2]
            else:
                dim_x, dim_y = [tx_0, cx_1 // r, r], [cy_1, ty_1]
                p_0_i, p_0_j = [1], [1]
                p_1_i, p_1_j = [1, 2], [2, 0]
                y_i, y_j = [1, 2], [0, 1]


        rs_lengths = 'SEQ<1>'  # ? why
        hs_lengths = 'TUP<SEQ<' + ', '.join(map(str, dim_x)) + '>, SEQ<' + ', '.join(map(str, dim_y)) + '>>'
        ps_2_rhs_major = 'TUP<SEQ<' + ', '.join(map(str, p_0_i)) + '>, SEQ<' + ', '.join(map(str, p_1_i)) + '>>'
        ps_2_rhs_minor = 'TUP<SEQ<' + ', '.join(map(str, p_0_j)) + '>, SEQ<' + ', '.join(map(str, p_1_j)) + '>>'
        ys_2_rhs_major = 'SEQ<' +  ', '.join(map(str, y_i)) + '>'
        ys_2_rhs_minor = 'SEQ<' +  ', '.join(map(str, y_j)) + '>'

        return 'STDENC<' + \
                ', '.join([rs_lengths, hs_lengths, ps_2_rhs_major, ps_2_rhs_minor, ys_2_rhs_major, ys_2_rhs_minor]) + '>'



class gemm_gld_tile_enc_gen(object):
    # divide s/r dim into 2 sub dim to better control
    def __init__(self, b, s, r):
        self.b = b
        self.s = s
        self.r = r

    def serialize(self, layout):
        # generalize into x*y dim, x0x1, y0y1, where y is fast changing dim
        ty_array = [32, 24, 16, 12, 8, 6, 5, 4, 3, 2, 1]
        vectors = [32, 16, 8, 4, 2, 1]

        dim_x = self.s if layout == GemmLayout.S_R else self.r
        dim_y = self.r if layout == GemmLayout.S_R else self.s

        result_desc_array = []

        for ty in ty_array:
            if dim_y % ty != 0:
                continue
            cy = dim_y // ty
            if self.b % cy != 0:
                continue
            cx = self.b // cy
            if dim_x % cx != 0:
                continue
            tx = dim_x // cx

            # next we unmerge x, y into x0*x1, y0*y1
            found_vector = 0
            for v in vectors:
                if ty % v == 0:
                    found_vector = v
                    break
            if found_vector == 0:
                continue

            tx_0, tx_1, ty_0, ty_1 = tx,  1, ty//found_vector, found_vector
            cx_0, cx_1, cy_0, cy_1 = 1 , cx,             1   , cy

            assert tx_1 == 1 and cx_0 == 1 and cy_0== 1, 'no need to split further'

            result_desc_array.append(gemm_thread_group_desc(self.b, [tx_0, tx_1, ty_0, ty_1], [cx_0, cx_1, cy_0, cy_1], layout, WAVE_SIZE))
        return result_desc_array

class desc_emitter(object):
    def __init__(self, file_name):
        self.fp = None
        try:
            self.fp = open(file_name, "w")
        except IOError as e:
            print("can't open file:{}({})".format(file_name, e))
            sys.exit()

        self.fp.write(f'namespace ck::tile_program {{\n')
        self.fp.write('// clang-format off\n')
        self.fp.write(f'// generated by {os.path.basename(__file__)}\n')
        self.fp.write(f'// do not include this file directly\n')
        self.fp.write('template<index_t dim_x, index_t dim_y, index_t block_size, index_t vector, index_t wave_size = 64>\n')
        self.fp.write('struct gemm_global_load_tile_encoding_dispatch;\n')
        self.fp.write('\n')
        self.fp.write('template<typename layout, index_t dim_r, index_t dim_c, index_t block_size, index_t vector, index_t wave_size = 64>\n')
        self.fp.write('struct gemm_global_load_tile_encoding_dispatch_with_layout;\n')
        self.fp.write('\n')
        self.fp.write('template<index_t dim_r, index_t dim_c, index_t block_size, index_t vector, index_t wave_size = 64>\n')
        self.fp.write('struct gemm_global_load_tile_encoding_dispatch_with_layout<ck::tensor_layout::gemm::RowMajor, dim_r, dim_c, block_size, vector, wave_size>\n')
        self.fp.write('    { using type = typename gemm_global_load_tile_encoding_dispatch<dim_r, dim_c, block_size, vector, wave_size>::type; };\n')
        self.fp.write('\n')
        self.fp.write('template<index_t dim_r, index_t dim_c, index_t block_size, index_t vector, index_t wave_size = 64>\n')
        self.fp.write('struct gemm_global_load_tile_encoding_dispatch_with_layout<ck::tensor_layout::gemm::ColMajor, dim_r, dim_c, block_size, vector, wave_size>\n')
        self.fp.write('    { using type = typename gemm_global_load_tile_encoding_dispatch<dim_c, dim_r, block_size, vector, wave_size>::type; };\n')
        self.fp.write('\n')
        self.fp.write('#define STDENC StaticTensorDistributionEncoding\n')
        self.fp.write('#define SEQ Sequence\n')
        self.fp.write('#define TUP Tuple\n')
        self.fp.write('\n')
        self.unique_alias_set = set()
        self.auto_vector_desc_dict = dict()

    def record_auto_vector(self, desc):
        # this is the max vector size under certain tile size
        name = f'{desc.dim_x}x{desc.dim_y}_b{desc.size}'
        if name in self.auto_vector_desc_dict:
            if self.auto_vector_desc_dict[name].t_len[3] < desc.t_len[3]:
                self.auto_vector_desc_dict[name] = desc
        else:
            self.auto_vector_desc_dict[name] = desc

    # def emit_auto_vector_desc(self):
    #     for name, desc in self.auto_vector_desc_dict.items():
    #         alias_name = f'gemm_global_load_tile_encoding_{desc.dim_x}x{desc.dim_y}_b{desc.size}_va'
    #         self.fp.write(f'using {alias_name:<40} = ' + \
    #             f'thread_group_desc<{desc.size:3}, seq<{desc.dim_x:3},{desc.dim_y:3}>, ' + \
    #             f'seq<{desc.c_len[0]:3},{desc.c_len[1]:3},{desc.c_len[2]:3},{desc.c_len[3]:3}>, ' + \
    #             f'seq<{desc.t_len[0]:3},{desc.t_len[1]:3},{desc.t_len[2]:3},{desc.t_len[3]:3}>>;    ')
    #         self.fp.write(f'template<> struct gemm_global_load_tile_encoding_dispatch<{desc.dim_x:3}, {desc.dim_y:3}, {desc.size:3}, TLA_AUTO_VECTOR>'
    #                       f' {{ using type = {alias_name:<40}; }};')
    #         self.fp.write('\n')

    def emit(self, desc):
        
        alias_name = f'gemm_global_load_tile_encoding_{desc.dim_x}x{desc.dim_y}_b{desc.size}_v{desc.t_len[3]}'
        if alias_name in self.unique_alias_set:
            return
        self.unique_alias_set.add(alias_name)

        enc_str = desc.to_enc_str()

        self.fp.write(f'using {alias_name:<48} = {enc_str}; ')
        self.fp.write(f'template<> struct gemm_global_load_tile_encoding_dispatch<{desc.dim_x:3}, {desc.dim_y:3}, {desc.size:3}, {desc.t_len[3]:2}>'
                        f' {{ using type = {alias_name:<48}; }};')
        self.fp.write('\n')

        self.record_auto_vector(desc)

    def __del__(self):
        if self.fp != None:
            self.fp.write('// clang-format on\n')
            self.fp.write('#undef STDENC\n')
            self.fp.write('#undef SEQ\n')
            self.fp.write('#undef TUP\n')
            self.fp.write('} // namespace\n')
            self.fp.close()

def emit_dispatch(fp, dim_x, dim_y, block_size, vector, alias_name):
    fp.write(f'template<> gemm_global_load_tile_encoding_dispatch<{dim_x:3}, {dim_y:3}, {block_size:3}, {vector:2}> {{ using type = {alias_name:<48}; }};')

def emit_unique_desc(fp, desc):
    alias_name = f'gemm_thread_group_desc_{desc.dim_x}x{desc.dim_y}_b{desc.size}_v{desc.t_len[3]}'
    fp.write(f'using {alias_name:<40} = ' + \
            f'thread_group_desc<{desc.size:3}, seq<{desc.dim_x:3},{desc.dim_y:3}>, ' + \
            f'seq<{desc.c_len[0]:3},{desc.c_len[1]:3},{desc.c_len[2]:3},{desc.c_len[3]:3}>, ' + \
            f'seq<{desc.t_len[0]:3},{desc.t_len[1]:3},{desc.t_len[2]:3},{desc.t_len[3]:3}>>;    ')
    emit_dispatch(fp, desc.dim_x, desc.dim_y, desc.size, desc.t_len[3], alias_name)
    fp.write('\n')

def valid_desc(m, b, s, r, desc):
    dim_x = s if m == GemmLayout.S_R else r
    dim_y = r if m == GemmLayout.S_R else s
    assert dim_x == desc.dim_x and dim_y == desc.dim_y
    assert dim_x == desc.c_len[0] * desc.c_len[1] * desc.t_len[0] * desc.t_len[1]
    assert dim_y == desc.c_len[2] * desc.c_len[3] * desc.t_len[2] * desc.t_len[3]
    assert b == desc.c_len[0] * desc.c_len[1] * desc.c_len[2] * desc.c_len[3]

def gen(file_name, m_array = M_ARRAY, b_array=B_ARRAY, s_array=S_ARRAY, r_array=R_ARRAY):
    e = desc_emitter(file_name)
    flag=True
    for m, b, s, r in itertools.product(m_array, b_array, s_array, r_array):
        co = gemm_gld_tile_enc_gen(b, s, r)
        desc_array = co.serialize(m)
        for desc in desc_array:
            valid_desc(m, b, s, r, desc)
            e.emit(desc)

                #flag = False
    # e.emit_auto_vector_desc()

if __name__ == '__main__':
    output_file = OUTPUT_FILE
    if len(sys.argv) >= 2:
        output_file = sys.argv[1]
    gen(output_file)
