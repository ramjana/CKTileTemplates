import enum
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
import os

WAVE_SIZE = 64
BANKS = 32
NO_TOUCH = WAVE_SIZE + 1

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

layout_tag = {
    layout_t.row_major : 'row',
    layout_t.col_major : 'col'
}

class inst_mfma_t(object):
    def __init__(self, a_type, b_type, c_type,  m, n, k, blocks, regs_a, regs_b, regs_c, wave_size = WAVE_SIZE):
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

    def name(self):
        return f"{self.m}x{self.n}x{self.k}_{data_type_tag[self.a_type]}"

D=data_type_t
#                                    a_type  b_type  c_type  m   n   k    b  va vb vc 
v_mfma_f32_16x16x16f16 = inst_mfma_t(D.fp16, D.fp16, D.fp32, 16, 16, 16,  1, 2, 2, 16)
v_mfma_f32_32x32x8f16  = inst_mfma_t(D.fp16, D.fp16, D.fp32, 32, 32,  8,  1, 2, 2,  4)


class grouped_xor_t(object):
    def __init__(self, elems_per_group, elems_per_row):
        self.elems_per_group = elems_per_group
        self.elems_per_row = elems_per_row
        self.groups_per_row = elems_per_row // elems_per_group
        # print(f'elems_per_group:{self.elems_per_group}, elems_per_row:{self.elems_per_row}, groups_per_row:{self.groups_per_row}')

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

    def idx_to_2d(self, linear_idx):
        i_col = linear_idx % self.elems_per_row
        i_row = linear_idx // self.elems_per_row
        return i_row, i_col

    def idx_to_2d_as_group(self, linear_idx):
        i_col = linear_idx % self.elems_per_row
        i_row = linear_idx // self.elems_per_row
        i_group = i_col // self.elems_per_group
        return i_row, i_group

# TODO: this is for a matrix
class shared_block_desc_t(object):
    def __init__(self, m_per_block, k_per_block, layout, k_pack = -1):
        self.m_per_block = m_per_block
        self.k_per_block = k_per_block
        self.layout = layout
        self.k_pack = k_pack

    def calculate_offset(self, coord_m_k):
        assert coord_m_k[0] < self.m_per_block and coord_m_k[1] < self.k_per_block, f'coord:{coord_m_k}, mxk:{m_per_block},{k_per_block}'
        if self.layout == layout_t.row_major:
            # m * k
            return coord_m_k[0] * k_per_block + coord_m_k[1]
        else:
            # k0 * m * k1
            return (coord_m_k[1] // self.k_pack) * self.m_per_block * self.k_pack + \
                    coord_m_k[0] * self.k_pack + coord_m_k[1] % self.k_pack

    def get_bounding_rec_sld(self, grouped_xor, mfma):
        # calculate what's the bounding rectangular size if need use a specific mfma, and do shared load
        if self.layout == layout_t.row_major:
            ks_per_row = grouped_xor.elems_per_row / self.k_per_block
            rows = int(mfma.m / ks_per_row)
            cols = grouped_xor.groups_per_row
        else:
            ms_per_row = grouped_xor.groups_per_row
            #rows = int((self.k_per_block / grouped_xor.elems_per_group) * (mfma.m / ms_per_row))
            rows = (self.m_per_block // ms_per_row) * (mfma.wave_size // mfma.m)
            cols = grouped_xor.groups_per_row
        return rows, cols

def plot_banks(x_coord, y_coord, values, elem_limit, elem_total, title, fig_name):
    def get_p_colors_0(i):
        # p_colors = ["paleturquoise", "lightcyan", "lightskyblue", "lightsteelblue",
        #     "peachpuff", "lightcoral", "sandybrown", "lightpink",
        #     "honeydew", "lightgreen", "aquamarine", "palegreen",
        #     "lemonchiffon", "moccasin", "khaki", "lightgoldenrodyellow"]
        p_colors = ["paleturquoise", "lightcyan"]
        return p_colors[i % len(p_colors)]
    def get_p_colors_1(i):
        p_colors = ["moccasin", "lightgoldenrodyellow"]
        return p_colors[i % len(p_colors)]
    if True:
        fontsize = 4
        cmap = matplotlib.colors.ListedColormap( [ ( get_p_colors_0(i) if i < elem_limit else get_p_colors_1(i) if i < 2 *elem_limit else "white") for i in range(elem_total + 1) ] )
        fig, ax = plt.subplots()
        ax.set_aspect(1) # Y/X ratio
        ax.invert_yaxis() # now Y is from top to bottom
        ax.set_title(title)
        #ax.set_axis_off() # turn off axes
        #plt.pcolormesh(x_coord, y_coord, values, cmap=cmap, edgecolors='k', linewidths=1)
        plt.pcolormesh(x_coord, y_coord, values, cmap=cmap, edgecolors='k')
        plt.xticks(x_coord, fontsize=fontsize)
        plt.yticks(y_coord, fontsize=fontsize)

        #fig.tight_layout()
        for ix in range(len(x_coord) - 1):
            for iy in range(len(y_coord) - 1):
                v = values[iy, ix]
                vstr = "X" if v == NO_TOUCH else f't{v}'
                text = ax.text((x_coord[ix] + x_coord[ix+1]) / 2,(y_coord[iy] + y_coord[iy+1])/2, vstr, fontsize=fontsize, ha="center", va="center")
        plt.savefig(fig_name, dpi=300)
        plt.close() # otherwise will have more than 20 figs open, will complain

    else:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_aspect(1) # Y/X ratio
        table = ax.table(
            cellText=values,
            rowLabels=y_coord,
            colLabels=x_coord,
            rowColours=["palegreen"] * 16,
            colColours=["palegreen"] * 16,
            #cellColours=[[".95" for c in range(16)] for r in range(16)],
            cellLoc='center',
            loc='upper left',
        )
        # for key, cell in table.get_celld().items():
        #     row, col = key
        #     if row > 0 and col > -1:  # Beware of table's idiosyncratic indexing...
        #         v = values[row, col]
        #         vstr = "X" if v == NO_TOUCH else f't{v}'
        #         # cell.set_text_props(vstr)
        plt.savefig(fig_name, dpi=800)

class bank_conflict_validator_t(object):
    def __init__(self, wave_size, banks, d_type, grouped_xor,  bounding_rec_sld = None, bounding_rec_sst = None):
        self.wave_size = wave_size
        self.banks = banks
        self.d_type = d_type
        self.grouped_xor = grouped_xor

        vector_bytes = grouped_xor.elems_per_group * (32 // data_type_bits[d_type])
        vector_dwords = (vector_bytes + 3) // 4
        self.mesh_x = banks // vector_dwords # how many threads to check the bank conflict
        assert self.wave_size % self.mesh_x == 0
        self.mesh_y = self.wave_size // self.mesh_x
        self.mesh = np.zeros((self.mesh_y, self.mesh_x), dtype=np.int32)

        sld_rows, sld_cols = bounding_rec_sld
        self.offsets = np.full((sld_rows, sld_cols), NO_TOUCH, dtype=np.int32)
        # print(f'@@@@ shape:{self.offsets.shape}')

    def enqueue(self, tid, group, offset):
        y = tid // self.mesh_x
        x = tid % self.mesh_x
        self.mesh[y][x] = group
        #print(f'tid:{tid}, x:{x}, y:{y}, g:{group}')
        #print(self.mesh)
        i_row, i_group = self.grouped_xor.idx_to_2d_as_group(offset)
        print(f'addr:{i_row}x{i_group}, shape:{self.offsets.shape}, tid:{tid}, os:{offset}', flush=True)
        self.offsets[i_row, i_group] = tid

    def valid_and_gen(self, base_dir, label = "test"):
        for g in self.mesh:
            msg = 'bank conflict free' if np.unique(g).size == g.size else 'has bank conflict'
            print(f'{g}, {msg}')

        # construct coord
        x_coord = np.array([i * self.grouped_xor.elems_per_group for i in range(self.offsets.shape[1] +1 )], dtype=np.int32)
        y_coord = np.array([i for i in range(self.offsets.shape[0] +1 )], dtype=np.int32)
        # print(f'x:{x_coord}, y:{y_coord}')

        plot_banks(x_coord, y_coord, self.offsets, self.offsets.shape[1], self.wave_size, label, os.path.join(base_dir, label+".png"))

def try_create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

# TODO: this is for a matrix. B matrix is the same
class simulator_t(object):
    SLD_BASE_FOLDER = 'sld'
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

        assert (self.k_pack % mfma.elem_a == 0)
        self.num_mfma_per_sld = self.k_pack // mfma.elem_a

        self.shared_block_desc = shared_block_desc_t(m_per_block, k_per_block, layout, self.k_pack)

        elems_per_group = self.k_pack
        elems_per_row = self.banks * (32 // data_type_bits[d_type])
        self.grouped_xor = grouped_xor_t(elems_per_group, elems_per_row)

        bounding_rec_sld = self.shared_block_desc.get_bounding_rec_sld(self.grouped_xor, mfma)
        self.sld_bank_conflict_validator = bank_conflict_validator_t(mfma.wave_size, banks, d_type, self.grouped_xor, bounding_rec_sld)

        try_create_dir(self.SLD_BASE_FOLDER)

    def sld(self):
        label = f'sld_m{self.m_per_block}_k{self.k_per_block}_v{self.k_pack}_{data_type_tag[self.d_type]}_{layout_tag[self.layout]}_{self.mfma.name()}'
        print(f'generating {label}')
        if self.num_mfma_per_sld * self.mfma.elem_a * self.mfma.k_groups_a > self.k_per_block:
            print(f'{self.num_mfma_per_sld } issues mfma per single sld can not cover whole k_per_block:{self.k_per_block}, ignore')
            return
        for tid in range(self.mfma.wave_size):
            idx_m = tid % self.mfma.m
            idx_k = tid // self.mfma.m * self.k_pack

            coord = [idx_m, idx_k]
            offset = self.shared_block_desc.calculate_offset(coord)
            offset_xor = self.grouped_xor.cvt(offset)
            group = self.grouped_xor.group_of_idx(offset_xor)
            print(f'[{tid:3}], coord:{coord}, offset:{offset}, offset_xor:{offset_xor}, group:{group},', end =" ", flush=True)
            self.sld_bank_conflict_validator.enqueue(tid, group, offset_xor)

        self.sld_bank_conflict_validator.valid_and_gen(self.SLD_BASE_FOLDER, label)

    def sst(self):
        pass

if __name__ == '__main__':

    M_BLOCKS = [32, 64, 96, 128, 160, 192, 256]
    K_BLOCKS = [8, 16, 32, 64]
    LAYOUT = [layout_t.row_major, layout_t.col_major]
    DTYPE = [data_type_t.fp16]
    MFMA = [v_mfma_f32_32x32x8f16, v_mfma_f32_16x16x16f16]
    K_PACK = ['origin', 'large']

    def get_kpack(k_pack, mfma):
        if k_pack == 'origin':
            return mfma.elem_a
        elif k_pack == 'large':
            return 4 * (32 // data_type_bits[mfma.a_type])
        else:
            assert False
            return 0
    for m_per_block, k_per_block, layout, d_type, mfma, k_pack in itertools.product(M_BLOCKS, K_BLOCKS, LAYOUT, DTYPE, MFMA, K_PACK):
        k_pack_ = get_kpack(k_pack, mfma)
        sim = simulator_t(m_per_block, k_per_block, layout, d_type, mfma, k_pack=k_pack_)
        sim.sld()
