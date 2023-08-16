import enum
import numpy as np

BANKS = 32

class data_type_t(enum.Enum):
    fp32 = enum.auto()
    fp16 = enum.auto()
    bf16 = enum.auto()
    int8 = enum.auto()

data_type_bits = {
    data_type_t.fp32 : 32,
    data_type_t.fp16 : 16,
    data_type_t.bf16 : 16,
    data_type_t.int8 : 8
}

data_type_tag = {
    data_type_t.fp32 : 'fp32',
    data_type_t.fp16 : 'fp16',
    data_type_t.bf16 : 'bf16',
    data_type_t.int8 : 'int8'
}

class layout_t(enum.Enum):
    row_major = enum.auto()
    col_major = enum.auto()

class inst_mfma_t(object):
    def __init__(self, a_type, b_type, c_type,  m, n, k, blocks, regs_a, regs_b, regs_c, wave_size = 64):
        self.a_type = a_type
        self.b_type = b_type
        self.c_type = c_type
        self.m = m
        self.n = n
        self.k = k
        self.blocks = blocks
        self.regs_a = regs_a
        self.regs_b = regs_b
        self.regs_c = regs_c
        self.wave_size = wave_size

        # TODO: double?
        self.elem_a = regs_a * (32 // data_type_bits[a_type])
        self.elem_b = regs_b * (32 // data_type_bits[b_type])
        self.elem_c = regs_c * (32 // data_type_bits[c_type])

        # TODO: only valid for single block
        self.k_groups_a = k // self.elem_a
        self.k_groups_b = k // self.elem_b

D=data_type_t
#                                    a_type  b_type  c_type  m   n   k    b  va vb vc 
v_mfma_f32_16x16x16f16 = inst_mfma_t(D.fp16, D.fp16, D.fp32, 16, 16, 16,  1, 2, 2, 16)
v_mfma_f32_32x32x8f16  = inst_mfma_t(D.fp16, D.fp16, D.fp32, 32, 32,  8,  1, 2, 2,  4)


class grouped_xor_t(object):
    def __init__(self, elems_per_group, elems_per_row):
        self.elems_per_group = elems_per_group
        self.elems_per_row = elems_per_row
        self.groups_per_row = elems_per_row // elems_per_group
        print(f'elems_per_group:{self.elems_per_group}, elems_per_row:{self.elems_per_row}, groups_per_row:{self.groups_per_row}')

    def cvt(self, linear_idx):
        i_col = linear_idx % self.elems_per_row
        i_row = linear_idx // self.elems_per_row
        i_group = i_col // self.elems_per_group
        xor_factor = i_row % self.groups_per_row
        i_xor_group = i_group ^ xor_factor
        new_idx = i_row * self.elems_per_row + i_xor_group *  self.elems_per_group
        return new_idx

    def group_of_idx(self, linear_idx):
        assert linear_idx % self.elems_per_group == 0
        i_col = linear_idx % self.elems_per_row
        i_group = i_col // self.elems_per_group
        return i_group

# TODO: this is for a matrix
class shared_block_desc_t(object):
    def __init__(self, m_per_block, k_per_block, layout, k_pack = -1):
        self.m_per_block = m_per_block
        self.k_per_block = k_per_block
        self.layout = layout
        self.k_pack = k_pack

    def calculate_offset(self, coord_m_k):
        assert(coord_m_k[0] < m_per_block and coord_m_k[1] < k_per_block)
        if self.layout == layout_t.row_major:
            # m * k
            return coord_m_k[0] * k_per_block + coord_m_k[1]
        else:
            # k0 * m * k1
            return (coord_m_k[1] // self.k_pack) * self.m_per_block * self.k_pack + \
                    coord_m_k[0] * self.k_pack + coord_m_k[1] % self.k_pack

class bank_conflict_validator_t(object):

    def __init__(self, wave_size, banks, d_type, grouped_xor):
        self.wave_size = wave_size
        self.banks = banks
        self.d_type = d_type
        self.grouped_xor = grouped_xor

        vector_bytes = grouped_xor.elems_per_group * (32 // data_type_bits[d_type])
        vector_dwords = (vector_bytes + 3) // 4
        self.mesh_y = banks // vector_dwords # how many threads to check the bank conflict
        assert self.wave_size % self.mesh_y == 0
        self.mesh_x = self.wave_size // self.mesh_y
        self.mesh = np.zeros((self.mesh_x, self.mesh_y), dtype=np.int32)

    def enqueue(self, tid, group):
        x = tid // self.mesh_y
        y = tid % self.mesh_y
        self.mesh[x][y] = group
        #print(f'tid:{tid}, x:{x}, y:{y}, g:{group}')
        #print(self.mesh)

    def valid(self):
        for g in self.mesh:
            msg = 'bank conflict free' if np.unique(g).size == g.size else 'has bank conflict'
            print(f'{g}, {msg}')


# TODO: this is for a matrix. B matrix is the same
class simulator_t(object):
    def __init__(self, m_per_block, k_per_block, layout, d_type, mfma, banks = BANKS, k_pack = -1):
        self.m_per_block = m_per_block
        self.k_per_block = k_per_block
        self.layout = layout
        self.d_type = d_type
        self.mfma = mfma
        self.banks = banks
        if k_pack == -1:
            self.k_pack = mfma.elem_a # only consider A
        else:
            self.k_pack = k_pack

        self.shared_block_desc = shared_block_desc_t(m_per_block, k_per_block, layout, self.k_pack)

        elems_per_group = self.k_pack
        elems_per_row = self.banks * (32 // data_type_bits[d_type])
        self.grouped_xor = grouped_xor_t(elems_per_group, elems_per_row)
        self.sld_bank_conflict_validator = bank_conflict_validator_t(mfma.wave_size, banks, d_type, self.grouped_xor)

    def sld(self):
        for tid in range(self.mfma.wave_size):
            idx_m = tid % self.mfma.m
            idx_k = tid // self.mfma.m * self.mfma.elem_a

            coord = [idx_m, idx_k]
            offset = self.shared_block_desc.calculate_offset(coord)
            offset_xor = self.grouped_xor.cvt(offset)
            group = self.grouped_xor.group_of_idx(offset_xor)
            print(f'[{tid:3}], coord:{coord}, offset:{offset}, offset_xor:{offset_xor}, group:{group}')
            self.sld_bank_conflict_validator.enqueue(tid, group)
        self.sld_bank_conflict_validator.valid()


    def sst(self):
        pass

if __name__ == '__main__':
    m_per_block = 128
    k_per_block = 32
    layout = layout_t.row_major
    d_type = data_type_t.fp16
    mfma = v_mfma_f32_32x32x8f16
    # mfma = v_mfma_f32_16x16x16f16
    sim = simulator_t(m_per_block, k_per_block, layout, d_type, mfma, k_pack=8)

    sim.sld()