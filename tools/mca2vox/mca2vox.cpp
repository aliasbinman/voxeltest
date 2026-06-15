// mca2vox.cpp — convert a Minecraft 1.18+ Java world (Anvil .mca region
// files) into a single .vox (VXL3). Whole world is scanned; only blocks
// listed in the hardcoded palette are emitted. VisMask is computed
// against in-section neighbours only (section boundary faces always
// visible — slight overdraw, no correctness loss). AO is left zero.
//
// Usage:
//   mca2vox.exe <world_dir> <out.vox>
//   mca2vox.exe "C:\dev\VoxelAssets\Sulfuria - 4000(1.21.5)" assets\sulfuria.vox

#define _CRT_SECURE_NO_WARNINGS
#include "asset_version.h"
#include "miniz.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <deque>
#include <utility>
#include <tuple>

namespace fs = std::filesystem;

// ---------- .vox writer types ----------
#pragma pack(push, 1)
struct DiskVoxel
{
    uint8_t  x, y, z;
    uint8_t  visMask;
    uint16_t paletteIdx;
    uint8_t  aoPacked[3];
};
struct ChunkMeta
{
    uint16_t cx, cy, cz, _pad;
    uint32_t voxelCount;
    uint32_t voxelOffset;
};
#pragma pack(pop)
static_assert(sizeof(DiskVoxel) == 9, "");
static_assert(sizeof(ChunkMeta) == 16, "");

// ---------- hardcoded block palette ----------
struct BlockEntry { const char* name; uint32_t rgb; };
static const BlockEntry kBlocks[] = {
    {"stone",                0x808080},
    {"granite",              0x9C6651},
    {"polished_granite",     0xA77562},
    {"diorite",              0xCDCDC4},
    {"polished_diorite",     0xDADAD3},
    {"andesite",             0x868686},
    {"polished_andesite",    0x919191},
    {"deepslate",            0x4C4C4C},
    {"cobbled_deepslate",    0x5A5A5A},
    {"polished_deepslate",   0x595959},
    {"tuff",                 0x6D6E66},
    {"calcite",              0xE0E0DC},
    {"cobblestone",          0x707070},
    {"mossy_cobblestone",    0x657350},
    {"dirt",                 0x866043},
    {"coarse_dirt",          0x76502A},
    {"rooted_dirt",          0x8D6243},
    {"podzol",               0x5C3F1B},
    {"grass_block",          0x507A32},
    {"farmland",             0x5C3D1F},
    {"dirt_path",            0x9A7F3F},
    {"sand",                 0xDED4AD},
    {"red_sand",             0xA85024},
    {"gravel",               0x837F7E},
    {"clay",                 0xA0A7B3},
    {"water",                0x3F76E4},
    {"lava",                 0xCF5F19},
    {"ice",                  0x9FCFFB},
    {"packed_ice",           0x8FB7FB},
    {"blue_ice",             0x77A8F9},
    {"snow_block",           0xF9FAFA},
    {"snow",                 0xF9FAFA},
    {"powder_snow",          0xEEF1F5},
    {"oak_log",              0x6E572B},
    {"oak_wood",             0x6E572B},
    {"stripped_oak_log",     0xB58D55},
    {"oak_planks",           0xB18A55},
    {"oak_leaves",           0x3F7E2F},
    {"spruce_log",           0x4B331E},
    {"stripped_spruce_log",  0x866B41},
    {"spruce_planks",        0x735232},
    {"spruce_leaves",        0x2F4F2A},
    {"birch_log",            0xD7D3C2},
    {"stripped_birch_log",   0xC9B47B},
    {"birch_planks",         0xC2A572},
    {"birch_leaves",         0x6A9A48},
    {"jungle_log",           0x6B5333},
    {"stripped_jungle_log",  0xB18154},
    {"jungle_planks",        0xAB825E},
    {"jungle_leaves",        0x3D8A19},
    {"acacia_log",           0x97574A},
    {"acacia_planks",        0xAB5D33},
    {"acacia_leaves",        0x6E7F35},
    {"dark_oak_log",         0x3B2C18},
    {"dark_oak_planks",      0x432B12},
    {"dark_oak_leaves",      0x2C4720},
    {"cherry_log",           0xC58F8B},
    {"cherry_planks",        0xE2B9B1},
    {"cherry_leaves",        0xEAB1D3},
    {"mangrove_log",         0x5B342E},
    {"mangrove_planks",      0x753D2D},
    {"mangrove_leaves",      0x71B231},
    {"netherrack",           0x6B2825},
    {"crimson_nylium",       0x88291E},
    {"warped_nylium",        0x2A6E5E},
    {"soul_sand",            0x564135},
    {"soul_soil",            0x4D3A2E},
    {"basalt",               0x4B4949},
    {"smooth_basalt",        0x4B4949},
    {"blackstone",           0x2B2530},
    {"polished_blackstone",  0x342E37},
    {"end_stone",            0xDADBA8},
    {"obsidian",             0x14102A},
    {"crying_obsidian",      0x251A38},
    {"glowstone",            0xFFB13C},
    {"sea_lantern",          0xBED3CB},
    {"shroomlight",          0xE48A2E},
    {"glass",                0xA7D3F0},
    {"coal_ore",             0x474747},
    {"deepslate_coal_ore",   0x3D3D3D},
    {"iron_ore",             0xA17D60},
    {"deepslate_iron_ore",   0x7D6852},
    {"copper_ore",           0xB87333},
    {"deepslate_copper_ore", 0x8B5A2B},
    {"gold_ore",             0xFCEE4B},
    {"deepslate_gold_ore",   0xB6A732},
    {"diamond_ore",          0x5DECF5},
    {"deepslate_diamond_ore",0x4FB2BF},
    {"redstone_ore",         0xA52A2A},
    {"deepslate_redstone_ore",0x842323},
    {"lapis_ore",            0x21497B},
    {"deepslate_lapis_ore",  0x1B3A60},
    {"emerald_ore",          0x17DD62},
    {"deepslate_emerald_ore",0x12A24A},
    {"ancient_debris",       0x5A2E22},
    {"raw_iron_block",       0xA7846A},
    {"raw_copper_block",     0xA45F38},
    {"raw_gold_block",       0xDBB534},
    {"iron_block",           0xDADADA},
    {"gold_block",           0xFCEE4B},
    {"diamond_block",        0x5DECF5},
    {"emerald_block",        0x17DD62},
    {"lapis_block",          0x21497B},
    {"redstone_block",       0xAB1300},
    {"netherite_block",      0x39363B},
    {"moss_block",           0x596F2C},
    {"moss_carpet",          0x596F2C},
    {"azalea",               0x547435},
    {"flowering_azalea",     0x856A8F},
    {"azalea_leaves",        0x547435},
    {"flowering_azalea_leaves",0x855E91},
    {"mud",                  0x3A2E26},
    {"packed_mud",           0x7D5C40},
    {"mud_bricks",           0x866647},
    {"sandstone",            0xDBCEA4},
    {"smooth_sandstone",     0xDBCEA4},
    {"cut_sandstone",        0xD9CB9F},
    {"chiseled_sandstone",   0xDBCEA4},
    {"red_sandstone",        0xA84B1F},
    {"smooth_red_sandstone", 0xA84B1F},
    {"terracotta",           0x985E43},
    {"white_terracotta",     0xD1B1A1},
    {"orange_terracotta",    0xA15426},
    {"magenta_terracotta",   0x95576C},
    {"light_blue_terracotta",0x706C8A},
    {"yellow_terracotta",    0xB98423},
    {"lime_terracotta",      0x677535},
    {"pink_terracotta",      0xA24E4E},
    {"gray_terracotta",      0x392923},
    {"light_gray_terracotta",0x876B62},
    {"cyan_terracotta",      0x575B5B},
    {"purple_terracotta",    0x764656},
    {"blue_terracotta",      0x4A3B5B},
    {"brown_terracotta",     0x4D3324},
    {"green_terracotta",     0x4C532A},
    {"red_terracotta",       0x8E3C2E},
    {"black_terracotta",     0x251710},
    {"bricks",               0x96503A},
    {"stone_bricks",         0x7A7A7A},
    {"mossy_stone_bricks",   0x6F7A5E},
    {"cracked_stone_bricks", 0x6F6F6F},
    {"chiseled_stone_bricks",0x787878},
    {"nether_bricks",        0x2D161A},
    {"red_nether_bricks",    0x470F0D},
    {"end_stone_bricks",     0xDADBA8},
    {"prismarine",           0x69A498},
    {"prismarine_bricks",    0x5FA298},
    {"dark_prismarine",      0x355C4A},
    {"white_wool",           0xE9ECEC},
    {"orange_wool",          0xF07613},
    {"magenta_wool",         0xBD44B3},
    {"light_blue_wool",      0x3AAFD9},
    {"yellow_wool",          0xF8C627},
    {"lime_wool",            0x70B919},
    {"pink_wool",            0xED8DAC},
    {"gray_wool",            0x3E4447},
    {"light_gray_wool",      0x8E8E86},
    {"cyan_wool",            0x158991},
    {"purple_wool",          0x792AAC},
    {"blue_wool",            0x35399D},
    {"brown_wool",           0x724728},
    {"green_wool",           0x546D1B},
    {"red_wool",             0xA12722},
    {"black_wool",           0x141519},
    {"oak_fence",            0xB18A55},
    {"oak_stairs",           0xB18A55},
    {"oak_slab",             0xB18A55},
    {"stone_stairs",         0x808080},
    {"stone_slab",           0x808080},
    {"cobblestone_stairs",   0x707070},
    {"cobblestone_slab",     0x707070},
    {"glass_pane",           0xA7D3F0},
    {"bamboo",               0x819C40},
    {"bamboo_block",         0x819C40},
    {"hay_block",            0xA9881A},
    {"melon",                0xAFAB17},
    {"pumpkin",              0xBC6915},
    {"carved_pumpkin",       0xBC6915},
    {"jack_o_lantern",       0xC97A22},
    {"cactus",               0x576A2E},
    {"sugar_cane",           0xAAD477},
    {"tall_grass",           0x91BD59},
    {"large_fern",           0x5E7C2E},
};
static const int kBlockCount = (int)(sizeof(kBlocks) / sizeof(kBlocks[0]));

// ---------- big-endian readers ----------
struct R
{
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    bool need(size_t n) { if (p + n > end) { ok = false; return false; } return true; }
    uint8_t  u8()  { if (!need(1)) return 0; return *p++; }
    int8_t   i8()  { return (int8_t)u8(); }
    uint16_t u16() { if (!need(2)) return 0; uint16_t v = ((uint16_t)p[0] << 8) | p[1]; p += 2; return v; }
    int16_t  i16() { return (int16_t)u16(); }
    uint32_t u32() { if (!need(4)) return 0; uint32_t v = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; p += 4; return v; }
    int32_t  i32() { return (int32_t)u32(); }
    uint64_t u64() { if (!need(8)) return 0; uint64_t v = 0; for (int i = 0; i < 8; ++i) v = (v << 8) | p[i]; p += 8; return v; }
    int64_t  i64() { return (int64_t)u64(); }
    std::string str() {
        uint16_t n = u16();
        if (!need(n)) return {};
        std::string s((const char*)p, n);
        p += n;
        return s;
    }
    void skip(size_t n) { if (need(n)) p += n; }
};

enum {
    T_END = 0, T_BYTE = 1, T_SHORT = 2, T_INT = 3, T_LONG = 4,
    T_FLOAT = 5, T_DOUBLE = 6, T_BYTEARR = 7, T_STRING = 8,
    T_LIST = 9, T_COMPOUND = 10, T_INTARR = 11, T_LONGARR = 12
};

// Forward decls
static void SkipPayload(R& r, uint8_t type);
static void SkipCompound(R& r);
static void SkipList(R& r);

static void SkipPayload(R& r, uint8_t type)
{
    switch (type) {
    case T_BYTE:    r.skip(1); break;
    case T_SHORT:   r.skip(2); break;
    case T_INT:     r.skip(4); break;
    case T_LONG:    r.skip(8); break;
    case T_FLOAT:   r.skip(4); break;
    case T_DOUBLE:  r.skip(8); break;
    case T_BYTEARR: { int32_t n = r.i32(); r.skip((size_t)std::max(0, n)); break; }
    case T_STRING:  { (void)r.str(); break; }
    case T_LIST:    SkipList(r); break;
    case T_COMPOUND: SkipCompound(r); break;
    case T_INTARR:  { int32_t n = r.i32(); r.skip((size_t)std::max(0, n) * 4); break; }
    case T_LONGARR: { int32_t n = r.i32(); r.skip((size_t)std::max(0, n) * 8); break; }
    default: r.ok = false; break;
    }
}

static void SkipList(R& r)
{
    uint8_t t = r.u8();
    int32_t n = r.i32();
    for (int32_t i = 0; i < n && r.ok; ++i) SkipPayload(r, t);
}

static void SkipCompound(R& r)
{
    while (r.ok) {
        uint8_t t = r.u8();
        if (t == T_END) return;
        (void)r.str();
        SkipPayload(r, t);
    }
}

// ---------- per-section block extractor ----------
struct SectionData
{
    int8_t y = 0;
    std::vector<uint16_t> indices; // 4096 entries, palette index into local palette
    std::vector<std::string> palette;
};

// Decode block_states data LongArray into 4096 indices.
// 1.16+ packing: bits = max(4, ceil(log2(palLen))). Indices packed low-bit
// first; no spanning across longs.
static void DecodeBlockStates(const std::vector<uint64_t>& longs, int palLen,
                              std::vector<uint16_t>& out)
{
    out.assign(4096, 0);
    if (palLen <= 1) return;                       // single-block section
    int bits = 4;
    while ((1 << bits) < palLen) ++bits;
    int perLong = 64 / bits;
    uint64_t mask = (bits == 64) ? ~0ull : ((1ull << bits) - 1ull);
    int total = 4096;
    if ((int)longs.size() * perLong < total) {
        // malformed
        return;
    }
    int li = 0, slot = 0;
    for (int i = 0; i < total; ++i) {
        uint64_t v = (longs[li] >> (slot * bits)) & mask;
        out[i] = (uint16_t)v;
        if (++slot == perLong) { slot = 0; ++li; }
    }
}

// Pre-1.13 ("flattening") legacy numeric block ID -> modern block name. Data/Add
// nibbles (variants like wool colour, log type) are ignored — base block only.
// Returns nullptr for air / non-solid decoration (skipped). Unknown solids -> stone.
static const char* LegacyIdName(int id)
{
    switch (id) {
    case 0:   return nullptr;            // air
    case 1:   return "stone";
    case 2:   return "grass_block";
    case 3:   return "dirt";
    case 4:   return "cobblestone";
    case 5:   return "oak_planks";
    case 7:   return "stone";            // bedrock
    case 8: case 9:   return "water";
    case 10: case 11: return "lava";
    case 12:  return "sand";
    case 13:  return "gravel";
    case 14:  return "gold_ore";
    case 15:  return "iron_ore";
    case 16:  return "coal_ore";
    case 17:  return "oak_log";
    case 18:  return "oak_leaves";
    case 19:  return "stone";            // sponge
    case 20:  return "glass";
    case 21:  return "lapis_ore";
    case 22:  return "lapis_block";
    case 24:  return "sandstone";
    case 35:  return "white_wool";
    case 41:  return "gold_block";
    case 42:  return "iron_block";
    case 43:  return "stone_slab";       // double stone slab
    case 44:  return "stone_slab";
    case 45:  return "bricks";
    case 47:  return "oak_planks";       // bookshelf
    case 48:  return "mossy_cobblestone";
    case 49:  return "obsidian";
    case 52:  return "stone";            // spawner
    case 53:  return "oak_stairs";
    case 54: case 58: return "oak_planks"; // chest / crafting table
    case 56:  return "diamond_ore";
    case 57:  return "diamond_block";
    case 60:  return "farmland";
    case 61: case 62: return "stone";    // furnace
    case 73: case 74: return "redstone_ore";
    case 78:  return "snow";
    case 79:  return "ice";
    case 80:  return "snow_block";
    case 81:  return "cactus";
    case 82:  return "clay";
    case 83:  return "sugar_cane";
    case 84:  return "oak_planks";       // jukebox
    case 85:  return "oak_fence";
    case 86: case 91: return "pumpkin";
    case 87:  return "netherrack";
    case 88:  return "soul_sand";
    case 89:  return "glowstone";
    case 95:  return "glass";            // stained glass
    case 97:  return "stone";            // monster egg
    case 98:  return "stone_bricks";
    case 101: return "iron_block";       // iron bars
    case 102: return "glass_pane";
    case 103: return "melon";
    case 108: return "bricks";           // brick stairs
    case 109: return "stone_bricks";     // stone brick stairs
    case 110: return "grass_block";      // mycelium
    case 112: return "nether_bricks";
    case 114: return "nether_bricks";    // nether brick stairs
    case 121: return "end_stone";
    case 123: case 124: return "glowstone"; // redstone lamp
    case 125: return "oak_planks";       // double wood slab
    case 126: return "oak_slab";
    case 128: return "sandstone";        // sandstone stairs
    case 129: return "emerald_ore";
    case 133: return "emerald_block";
    case 134: return "spruce_log";       // spruce stairs (approx)
    case 135: return "birch_planks";
    case 136: return "jungle_planks";
    case 139: return "cobblestone";      // cobblestone wall
    case 152: return "redstone_block";
    case 155: return "diorite";          // quartz block (white-ish)
    case 156: return "diorite";          // quartz stairs
    case 159: return "white_terracotta"; // stained hardened clay
    case 161: return "acacia_leaves";    // leaves2 (acacia/dark oak)
    case 162: return "acacia_log";       // log2
    case 168: return "prismarine";
    case 169: return "sea_lantern";
    case 172: return "terracotta";       // hardened clay
    case 174: return "packed_ice";
    case 179: return "red_sandstone";
    case 180: return "red_sandstone";    // red sandstone stairs
    case 181: case 182: return "red_sandstone";
    case 201: case 206: return "end_stone_bricks"; // purpur / end brick (approx)
    // Non-solid decoration → skip (air).
    case 6: case 27: case 28: case 31: case 32: case 37: case 38: case 39:
    case 40: case 50: case 51: case 55: case 59: case 63: case 64: case 65:
    case 66: case 68: case 69: case 70: case 75: case 76: case 77: case 90:
    case 104: case 105: case 106: case 115: case 131: case 132: case 141:
    case 142: case 143: case 147: case 148: case 171:
        return nullptr;
    default:  return "stone";            // unknown solid → keep as stone
    }
}

// Pre-1.16 (DataVersion < 2529) packing: indices SPAN across 64-bit long
// boundaries (no per-long padding). Used by the old Level>Sections>BlockStates path.
static void DecodeBlockStatesSpanning(const std::vector<uint64_t>& longs, int palLen,
                                      std::vector<uint16_t>& out)
{
    out.assign(4096, 0);
    if (palLen <= 1) return;                       // single-block section (no array)
    int bits = 4;
    while ((1 << bits) < palLen) ++bits;
    uint64_t mask = (1ull << bits) - 1ull;
    long long need = (long long)4096 * bits;
    if ((long long)longs.size() * 64 < need) return; // malformed
    for (int i = 0; i < 4096; ++i) {
        long long bitpos = (long long)i * bits;
        int li  = (int)(bitpos >> 6);
        int off = (int)(bitpos & 63);
        uint64_t v = longs[li] >> off;
        if (off + bits > 64) v |= longs[li + 1] << (64 - off);
        out[i] = (uint16_t)(v & mask);
    }
}

// Parse one pre-1.18 section compound. Two sub-formats:
//   1.13–1.17: Palette(list<compound>) + BlockStates(LongArray, spanning).
//   pre-1.13 : Blocks(ByteArray 4096) + Add(ByteArray 2048 nibbles, opt) + Data(opt).
static void ParseOldSection(R& r, std::vector<SectionData>& outSecs)
{
    SectionData sd;
    bool haveY = false;
    std::vector<uint64_t> dataLongs;
    std::vector<uint8_t> blocks, addNib;
    int palLen = 0;
    while (r.ok) {
        uint8_t tt = r.u8();
        if (tt == T_END) break;
        std::string sn = r.str();
        if (tt == T_BYTE && sn == "Y") { sd.y = r.i8(); haveY = true; }
        else if (tt == T_LIST && sn == "Palette") {
            uint8_t plt = r.u8();
            int32_t pn = r.i32();
            if (plt != T_COMPOUND) { for (int j = 0; j < pn && r.ok; ++j) SkipPayload(r, plt); }
            else {
                palLen = pn;
                sd.palette.reserve(pn);
                for (int j = 0; j < pn && r.ok; ++j) {
                    std::string blockName;
                    while (r.ok) {
                        uint8_t pt = r.u8();
                        if (pt == T_END) break;
                        std::string pName = r.str();
                        if (pt == T_STRING && pName == "Name") blockName = r.str();
                        else SkipPayload(r, pt);
                    }
                    sd.palette.push_back(blockName);
                }
            }
        }
        else if (tt == T_LONGARR && sn == "BlockStates") {
            int32_t dn = r.i32();
            dataLongs.resize((size_t)std::max(0, dn));
            for (int32_t j = 0; j < dn && r.ok; ++j) dataLongs[j] = r.u64();
        }
        else if (tt == T_BYTEARR && sn == "Blocks") {
            int32_t dn = r.i32(); dn = std::max(0, dn);
            blocks.resize((size_t)dn);
            for (int32_t j = 0; j < dn && r.ok; ++j) blocks[j] = r.u8();
        }
        else if (tt == T_BYTEARR && sn == "Add") {
            int32_t dn = r.i32(); dn = std::max(0, dn);
            addNib.resize((size_t)dn);
            for (int32_t j = 0; j < dn && r.ok; ++j) addNib[j] = r.u8();
        }
        else SkipPayload(r, tt);
    }

    if (!blocks.empty()) {
        // Legacy numeric IDs → build a local name palette + indices.
        sd.indices.assign(4096, 0);
        std::unordered_map<int, uint16_t> idToLocal;
        sd.palette.push_back("air"); // local 0 = skipped by the name matcher
        auto nib = [&](const std::vector<uint8_t>& a, int i) -> int {
            if ((size_t)(i >> 1) >= a.size()) return 0;
            return (i & 1) ? (a[i >> 1] >> 4) : (a[i >> 1] & 0xF);
        };
        for (int i = 0; i < 4096 && i < (int)blocks.size(); ++i) {
            int id = blocks[i] | (nib(addNib, i) << 8);
            const char* nm = LegacyIdName(id);
            if (!nm) { sd.indices[i] = 0; continue; }     // air
            auto it = idToLocal.find(id);
            uint16_t li;
            if (it == idToLocal.end()) {
                li = (uint16_t)sd.palette.size();
                sd.palette.push_back(nm);
                idToLocal.emplace(id, li);
            } else li = it->second;
            sd.indices[i] = li;
        }
    } else {
        if (palLen <= 1) sd.indices.assign(4096, 0);     // single-block (no BlockStates)
        else             DecodeBlockStatesSpanning(dataLongs, palLen, sd.indices);
    }

    if (haveY && !sd.palette.empty())
        outSecs.push_back(std::move(sd));
}

// Parse one chunk's NBT payload. Extract sections into `outSecs`.
// Returns false on parse error.
static bool ParseChunkNbt(const uint8_t* nbt, size_t nbtLen,
                          std::vector<SectionData>& outSecs)
{
    R r{ nbt, nbt + nbtLen };
    // Root is named TAG_Compound
    uint8_t rootT = r.u8();
    if (rootT != T_COMPOUND) return false;
    (void)r.str(); // root name (usually "")

    // Walk root compound; find "sections" list (1.18+) or "Level" wrapper (pre-1.18).
    while (r.ok) {
        uint8_t t = r.u8();
        if (t == T_END) break;
        std::string name = r.str();
        if (t == T_COMPOUND && name == "Level") {
            // Pre-1.18: Level > Sections (capital) with Palette/BlockStates siblings.
            while (r.ok) {
                uint8_t lt = r.u8();
                if (lt == T_END) break;
                std::string ln = r.str();
                if (lt == T_LIST && ln == "Sections") {
                    uint8_t st = r.u8();
                    int32_t n = r.i32();
                    if (st != T_COMPOUND) { for (int i = 0; i < n && r.ok; ++i) SkipPayload(r, st); }
                    else { for (int i = 0; i < n && r.ok; ++i) ParseOldSection(r, outSecs); }
                } else {
                    SkipPayload(r, lt);
                }
            }
        }
        else if (t == T_LIST && name == "sections") {
            uint8_t lt = r.u8();
            int32_t n = r.i32();
            if (lt != T_COMPOUND) { for (int i = 0; i < n && r.ok; ++i) SkipPayload(r, lt); continue; }
            for (int i = 0; i < n && r.ok; ++i) {
                SectionData sd;
                bool haveY = false, haveBs = false;
                // section compound
                while (r.ok) {
                    uint8_t tt = r.u8();
                    if (tt == T_END) break;
                    std::string sn = r.str();
                    if (tt == T_BYTE && sn == "Y") { sd.y = r.i8(); haveY = true; }
                    else if (tt == T_COMPOUND && sn == "block_states") {
                        // inside block_states: palette (list of compound), data (LongArray)
                        std::vector<uint64_t> dataLongs;
                        int palLen = 0;
                        while (r.ok) {
                            uint8_t bt = r.u8();
                            if (bt == T_END) break;
                            std::string bn = r.str();
                            if (bt == T_LIST && bn == "palette") {
                                uint8_t plt = r.u8();
                                int32_t pn = r.i32();
                                if (plt != T_COMPOUND) {
                                    for (int j = 0; j < pn && r.ok; ++j) SkipPayload(r, plt);
                                } else {
                                    palLen = pn;
                                    sd.palette.reserve(pn);
                                    for (int j = 0; j < pn && r.ok; ++j) {
                                        // palette entry compound: { Name: string, Properties: compound? }
                                        std::string blockName;
                                        while (r.ok) {
                                            uint8_t pt = r.u8();
                                            if (pt == T_END) break;
                                            std::string pName = r.str();
                                            if (pt == T_STRING && pName == "Name") blockName = r.str();
                                            else SkipPayload(r, pt);
                                        }
                                        sd.palette.push_back(blockName);
                                    }
                                }
                            } else if (bt == T_LONGARR && bn == "data") {
                                int32_t dn = r.i32();
                                dataLongs.resize((size_t)std::max(0, dn));
                                for (int32_t j = 0; j < dn && r.ok; ++j) dataLongs[j] = r.u64();
                            } else {
                                SkipPayload(r, bt);
                            }
                        }
                        DecodeBlockStates(dataLongs, palLen, sd.indices);
                        haveBs = true;
                    } else {
                        SkipPayload(r, tt);
                    }
                }
                if (haveY && haveBs && !sd.palette.empty())
                    outSecs.push_back(std::move(sd));
            }
        } else {
            SkipPayload(r, t);
        }
    }
    return r.ok;
}

// Heuristic colour for an unknown block name (modded packs like WesterosCraft use
// thousands of custom blocks). Substring-match common materials so the geometry is
// kept (coloured roughly) instead of culled. Most-specific keywords first.
static bool FallbackColor(const char* s, uint32_t& outRgb)
{
    struct KW { const char* k; uint32_t rgb; };
    static const KW kw[] = {
        {"deepslate",0x4C4C4C}, {"cobble",0x707070}, {"sandstone",0xDBCEA4},
        {"red_sand",0xA85024},  {"sand",0xDED4AD},   {"stone_brick",0x7A7A7A},
        {"stonebrick",0x7A7A7A},{"nether_brick",0x2D161A}, {"brick",0x96503A},
        {"thatch",0xA9881A},    {"hay",0xA9881A},     {"straw",0xC9A227},
        {"dark_oak",0x432B12},  {"spruce",0x735232},  {"birch",0xC2A572},
        {"jungle",0xAB825E},    {"acacia",0xAB5D33},  {"mangrove",0x753D2D},
        {"oak",0xB18A55},       {"plank",0xB18A55},   {"firewood",0x6E572B},
        {"bark",0x5B4327},      {"log",0x6E572B},     {"wood",0x6E572B},
        {"leaves",0x3F7E2F},    {"leaf",0x3F7E2F},    {"foliage",0x3F7E2F},
        {"grass",0x507A32},     {"moss",0x596F2C},    {"fern",0x4A6A2A},
        {"mud",0x3A2E26},       {"bog",0x4A4030},     {"dirt",0x866043},
        {"soil",0x5C3F1B},      {"gravel",0x837F7E},  {"rock",0x808080},
        {"clay",0x985E43},      {"terracotta",0x985E43}, {"marble",0xDADAD3},
        {"snow",0xF9FAFA},      {"ice",0x9FCFFB},     {"water",0x3F76E4},
        {"glass",0xA7D3F0},     {"thin_",0x6E572B},
        {"wool",0xE9ECEC},      {"cloth",0xE9ECEC},   {"carpet",0xC8C8C8},
        {"leather",0x8A5A33},   {"fur",0xB7A07A},     {"wattle",0x9A7B4F},
        {"iron",0xDADADA},      {"gold",0xFCEE4B},    {"copper",0xB87333},
        {"fence",0x9A7B4F},     {"slab",0x9A8A6A},    {"stair",0x9A8A6A},
        {"sandstones",0xDBCEA4},{"stone",0x808080},
    };
    for (const auto& e : kw) if (strstr(s, e.k)) { outRgb = e.rgb; return true; }
    return false;
}

// ---------- block name → output palette index ----------
struct GlobalPal
{
    std::unordered_map<std::string, int16_t> nameToIdx;
    std::vector<uint32_t> rgb;            // 0xRRGGBB
    std::vector<std::string> unknown;     // names that matched nothing (skipped)
    int16_t Get(const std::string& fullName)
    {
        // strip "minecraft:" prefix
        const char* s = fullName.c_str();
        if (!strncmp(s, "minecraft:", 10)) s += 10;
        auto it = nameToIdx.find(s);
        if (it != nameToIdx.end()) return it->second;
        // search hardcoded table
        for (int i = 0; i < kBlockCount; ++i) {
            if (!strcmp(kBlocks[i].name, s)) {
                int16_t idx = (int16_t)rgb.size();
                rgb.push_back(kBlocks[i].rgb);
                nameToIdx.emplace(s, idx);
                return idx;
            }
        }
        // Skip only true air variants (not "stairs"/"chair" — exact components).
        if (!strcmp(s, "air") || strstr(s, "cave_air") || strstr(s, "void_air") ||
            strstr(s, "barrier") || strstr(s, "structure_void") || strstr(s, "light")) {
            nameToIdx.emplace(s, (int16_t)-1);
            return -1;
        }
        // Unknown solid (modded etc.) → keyword fallback colour so geo is kept.
        uint32_t fb = 0x808080;
        FallbackColor(s, fb);
        int16_t idx = (int16_t)rgb.size();
        rgb.push_back(fb);
        nameToIdx.emplace(s, idx);
        unknown.push_back(s);                // still logged so we can add real colours
        return idx;
    }
};

// ---------- region scan ----------
static bool DecompressZlib(const uint8_t* in, size_t inLen,
                           std::vector<uint8_t>& out)
{
    // Start with 4× estimate; grow on overflow.
    out.resize(std::max<size_t>(64 * 1024, inLen * 8));
    while (true) {
        mz_ulong outLen = (mz_ulong)out.size();
        int rc = mz_uncompress(out.data(), &outLen, in, (mz_ulong)inLen);
        if (rc == MZ_OK) { out.resize(outLen); return true; }
        if (rc == MZ_BUF_ERROR) { out.resize(out.size() * 2); continue; }
        return false;
    }
}

struct OutChunk
{
    std::vector<DiskVoxel> voxels;
};

// Per-mc-column dense solidity bitset for full Y range [byMin..byMax).
// Sized 16 (lx) × 16 (lz) × Ycount bits.
struct ColSolid
{
    std::vector<uint64_t> bits;
};

// Iterate all parseable chunks in all regions. cb(mcCX, mcCZ, sections).
template <class Cb>
static void ForEachChunk(const std::vector<struct RegionFile_>& regs, Cb cb);

struct RegionFile_ { fs::path path; int rx, rz; };

template <class Cb>
static void ForEachChunk(const std::vector<RegionFile_>& regs, Cb cb)
{
    std::vector<uint8_t> regBytes, nbtBuf;
    for (size_t ri = 0; ri < regs.size(); ++ri) {
        auto& reg = regs[ri];
        FILE* f = fopen(reg.path.string().c_str(), "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        size_t sz = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        regBytes.resize(sz);
        if (fread(regBytes.data(), 1, sz, f) != sz) { fclose(f); continue; }
        fclose(f);
        if (sz < 8192) continue;

        for (int idx = 0; idx < 1024; ++idx) {
            const uint8_t* loc = &regBytes[idx * 4];
            uint32_t off = ((uint32_t)loc[0] << 16) | ((uint32_t)loc[1] << 8) | loc[2];
            uint32_t scnt = loc[3];
            if (off == 0 || scnt == 0) continue;
            size_t fileOff = (size_t)off * 4096;
            if (fileOff + 5 > sz) continue;
            const uint8_t* hdr = &regBytes[fileOff];
            uint32_t clen = ((uint32_t)hdr[0] << 24) | ((uint32_t)hdr[1] << 16)
                          | ((uint32_t)hdr[2] << 8)  | hdr[3];
            uint8_t cType = hdr[4];
            if (clen == 0 || clen > sz - fileOff - 4) continue;
            const uint8_t* payload = hdr + 5;
            size_t payloadLen = clen - 1;

            const uint8_t* nbtPtr = nullptr;
            size_t nbtLen = 0;
            if (cType == 2) {
                if (!DecompressZlib(payload, payloadLen, nbtBuf)) continue;
                nbtPtr = nbtBuf.data();
                nbtLen = nbtBuf.size();
            } else if (cType == 3) {
                nbtPtr = payload;
                nbtLen = payloadLen;
            } else {
                continue;
            }

            std::vector<SectionData> secs;
            if (!ParseChunkNbt(nbtPtr, nbtLen, secs)) continue;

            int cxInReg = idx & 31;
            int czInReg = idx >> 5;
            int mcCX = reg.rx * 32 + cxInReg;
            int mcCZ = reg.rz * 32 + czInReg;
            cb(mcCX, mcCZ, secs, ri);
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: mca2vox <world_dir> <out.vox> [--sky-cull] [--height-cull[=N]]\n");
        return 1;
    }
    const char* worldPath = argv[1];
    const char* outPath = argv[2];
    bool skyCull = false;
    bool heightCull = false;
    int  heightCullN = 10;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--sky-cull")) skyCull = true;
        else if (!strcmp(argv[i], "--height-cull")) heightCull = true;
        else if (!strncmp(argv[i], "--height-cull=", 14)) {
            heightCull = true;
            heightCullN = atoi(argv[i] + 14);
            if (heightCullN < 0) heightCullN = 0;
        }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1; }
    }
    if (heightCull) skyCull = false;     // height-cull overrides sky-cull
    const int   D = 64;                       // chunkDim of output

    fs::path regionDir = fs::path(worldPath) / "region";
    if (!fs::is_directory(regionDir)) {
        fprintf(stderr, "no region/ subdir under %s\n", worldPath);
        return 1;
    }

    // First sweep: list region files + their (rx,rz).
    std::vector<RegionFile_> regions;
    for (auto& de : fs::directory_iterator(regionDir)) {
        if (!de.is_regular_file()) continue;
        std::string name = de.path().filename().string();
        int rx, rz;
        if (sscanf(name.c_str(), "r.%d.%d.mca", &rx, &rz) == 2) {
            regions.push_back({ de.path(), rx, rz });
        }
    }
    if (regions.empty()) {
        fprintf(stderr, "no r.X.Z.mca files in %s\n", regionDir.string().c_str());
        return 1;
    }
    printf("[mca2vox] world=%s regions=%zu\n", worldPath, regions.size());

    // World AABB in block coords.
    int32_t bxMin = INT32_MAX, bxMax = INT32_MIN;
    int32_t bzMin = INT32_MAX, bzMax = INT32_MIN;
    for (auto& r : regions) {
        int32_t x0 = r.rx * 512, z0 = r.rz * 512;
        bxMin = std::min(bxMin, x0);
        bzMin = std::min(bzMin, z0);
        bxMax = std::max(bxMax, x0 + 512);
        bzMax = std::max(bzMax, z0 + 512);
    }
    const int32_t byMin = -64;
    const int32_t byMax = 320;

    // Output chunk grid dims (round-up).
    auto Ceil = [](int32_t v, int32_t d) { return (v + d - 1) / d; };
    int32_t gridX = Ceil(bxMax - bxMin, D);
    int32_t gridY = Ceil(byMax - byMin, D);
    int32_t gridZ = Ceil(bzMax - bzMin, D);
    printf("[mca2vox] bounds x[%d..%d] y[%d..%d] z[%d..%d] grid=%dx%dx%d\n",
           bxMin, bxMax, byMin, byMax, bzMin, bzMax, gridX, gridY, gridZ);
    if (gridX > 0xFFFF || gridY > 0xFFFF || gridZ > 0xFFFF) {
        fprintf(stderr, "grid too large for uint16 ChunkMeta\n");
        return 1;
    }

    auto ChunkKey = [&](int32_t cx, int32_t cy, int32_t cz) -> uint64_t {
        return ((uint64_t)(uint32_t)cx)
             | ((uint64_t)(uint32_t)cy << 20)
             | ((uint64_t)(uint32_t)cz << 40);
    };
    std::unordered_map<uint64_t, OutChunk> outChunks;

    GlobalPal pal;

    // ----- Pass 1: build global per-mc-column solidity bitset. -----
    const int yCount = byMax - byMin;           // e.g. 384
    const int bitsPerCol = 16 * 16 * yCount;
    const int u64PerCol = (bitsPerCol + 63) / 64;
    auto ColIdx = [&](int lx, int ly, int lz) { return (ly * 16 + lz) * 16 + lx; };

    std::unordered_map<uint64_t, ColSolid> globalSolid;
    auto McKey = [](int cx, int cz) -> uint64_t {
        return ((uint64_t)(uint32_t)cx) | ((uint64_t)(uint32_t)cz << 32);
    };

    size_t pass1Chunks = 0;
    ForEachChunk(regions, [&](int mcCX, int mcCZ, std::vector<SectionData>& secs, size_t ri) {
        ColSolid* col = nullptr;
        for (auto& sd : secs) {
            int32_t baseY = (int32_t)sd.y * 16;
            if (baseY < byMin || baseY + 16 > byMax) {
                // partial overlap handled per-voxel below
            }
            // Map palette names to keep/skip.
            std::vector<int16_t> localToGlobal(sd.palette.size());
            bool anyKept = false;
            for (size_t pi = 0; pi < sd.palette.size(); ++pi) {
                localToGlobal[pi] = pal.Get(sd.palette[pi]);
                if (localToGlobal[pi] >= 0) anyKept = true;
            }
            if (!anyKept) continue;

            for (int y = 0; y < 16; ++y) {
                int32_t wy = baseY + y;
                if (wy < byMin || wy >= byMax) continue;
                int ly = wy - byMin;
                for (int z = 0; z < 16; ++z)
                for (int x = 0; x < 16; ++x) {
                    int li = (y * 16 + z) * 16 + x;
                    int pi = sd.indices[li];
                    if (pi < 0 || pi >= (int)localToGlobal.size()) continue;
                    if (localToGlobal[pi] < 0) continue;
                    if (!col) {
                        col = &globalSolid[McKey(mcCX, mcCZ)];
                        col->bits.assign(u64PerCol, 0);
                    }
                    int ci = ColIdx(x, ly, z);
                    col->bits[ci >> 6] |= (uint64_t)1 << (ci & 63);
                }
            }
        }
        ++pass1Chunks;
        if ((ri & 7) == 0 && (pass1Chunks & 1023) == 0) {
            printf("[pass1] regions ~%zu/%zu  chunks=%zu  cols=%zu  pal=%zu\n",
                   ri + 1, regions.size(), pass1Chunks, globalSolid.size(), pal.rgb.size());
        }
    });
    printf("[pass1] done  chunks=%zu  solid-cols=%zu  pal=%zu  colMem~%.1f MB\n",
           pass1Chunks, globalSolid.size(), pal.rgb.size(),
           (double)globalSolid.size() * u64PerCol * 8.0 / (1024 * 1024));

    // ----- Helpers shared by pass 2 (and sky flood below). -----
    auto NormCol = [&](int& cx, int& cz, int& lx, int& lz) {
        if (lx < 0)        { cx -= 1; lx += 16; }
        else if (lx >= 16) { cx += 1; lx -= 16; }
        if (lz < 0)        { cz -= 1; lz += 16; }
        else if (lz >= 16) { cz += 1; lz -= 16; }
    };
    auto SolidAt = [&](int mcCX, int mcCZ, int lx, int ly, int lz) -> bool {
        if (ly < 0 || ly >= yCount) return false;
        NormCol(mcCX, mcCZ, lx, lz);
        auto it = globalSolid.find(McKey(mcCX, mcCZ));
        if (it == globalSolid.end()) return false;
        int ci = ColIdx(lx, ly, lz);
        return (it->second.bits[ci >> 6] >> (ci & 63)) & 1ull;
    };

    // ----- Sky flood + kept derivation. -----
    std::unordered_map<uint64_t, ColSolid> globalKept;
    if (skyCull) {
        // 1. Classify absent cols: BFS over col grid from outer shell through absent cols.
        int cxMin = INT32_MAX, cxMax = INT32_MIN;
        int czMin = INT32_MAX, czMax = INT32_MIN;
        for (auto& kv : globalSolid) {
            int cx = (int32_t)(kv.first & 0xFFFFFFFFu);
            int cz = (int32_t)(kv.first >> 32);
            cxMin = std::min(cxMin, cx); cxMax = std::max(cxMax, cx);
            czMin = std::min(czMin, cz); czMax = std::max(czMax, cz);
        }
        std::unordered_map<uint64_t, uint8_t> colAir;
        {
            std::deque<std::pair<int,int>> q;
            auto Try = [&](int cx, int cz) {
                if (globalSolid.find(McKey(cx, cz)) != globalSolid.end()) return;
                uint64_t k = McKey(cx, cz);
                if (colAir.emplace(k, 1).second) q.push_back({ cx, cz });
            };
            for (int cz = czMin - 1; cz <= czMax + 1; ++cz) {
                Try(cxMin - 1, cz); Try(cxMax + 1, cz);
            }
            for (int cx = cxMin - 1; cx <= cxMax + 1; ++cx) {
                Try(cx, czMin - 1); Try(cx, czMax + 1);
            }
            while (!q.empty()) {
                auto [cx, cz] = q.front(); q.pop_front();
                for (int d = 0; d < 4; ++d) {
                    const int dx[4] = { +1, -1, 0, 0 };
                    const int dz[4] = { 0, 0, +1, -1 };
                    int nx = cx + dx[d], nz = cz + dz[d];
                    if (nx < cxMin - 1 || nx > cxMax + 1) continue;
                    if (nz < czMin - 1 || nz > czMax + 1) continue;
                    Try(nx, nz);
                }
            }
            printf("[skycull] col-grid cx[%d..%d] cz[%d..%d] absent-air=%zu\n",
                   cxMin, cxMax, czMin, czMax, colAir.size());
        }
        auto AbsentColIsAir = [&](int cx, int cz) -> bool {
            return colAir.find(McKey(cx, cz)) != colAir.end();
        };

        // 2. Build exposed-air bitset per col via voxel-level BFS.
        std::unordered_map<uint64_t, ColSolid> globalExposed;
        globalExposed.reserve(globalSolid.size() * 2);
        printf("[skycull] expected exposed colMem ~%.1f MB\n",
               (double)globalSolid.size() * u64PerCol * 8.0 / (1024 * 1024));

        // Seed.
        struct QE { int cx, cz, lx, ly, lz; };
        std::deque<QE> q;
        for (auto& kv : globalSolid) {
            int cx = (int32_t)(kv.first & 0xFFFFFFFFu);
            int cz = (int32_t)(kv.first >> 32);
            ColSolid& s = kv.second;
            ColSolid& e = globalExposed[McKey(cx, cz)];
            if (e.bits.empty()) e.bits.assign(u64PerCol, 0);

            // Seed top-Y plane (ly = yCount-1).
            int topY = yCount - 1;
            for (int lz = 0; lz < 16; ++lz)
            for (int lx = 0; lx < 16; ++lx) {
                int ci = ColIdx(lx, topY, lz);
                if ((s.bits[ci >> 6] >> (ci & 63)) & 1ull) continue;
                e.bits[ci >> 6] |= (uint64_t)1 << (ci & 63);
                q.push_back({ cx, cz, lx, topY, lz });
            }
            // Seed lateral cells whose +X/-X/+Z/-Z neighbor col is absent-air.
            const int faceDir[4][2] = { { +1, 0 }, { -1, 0 }, { 0, +1 }, { 0, -1 } };
            for (int d = 0; d < 4; ++d) {
                int ncx = cx + faceDir[d][0];
                int ncz = cz + faceDir[d][1];
                if (globalSolid.find(McKey(ncx, ncz)) != globalSolid.end()) continue;
                if (!AbsentColIsAir(ncx, ncz)) continue;
                int axis = (faceDir[d][0] != 0) ? 0 : 1;     // 0=X face, 1=Z face
                int fixed = (axis == 0)
                            ? (faceDir[d][0] > 0 ? 15 : 0)
                            : (faceDir[d][1] > 0 ? 15 : 0);
                for (int ly = 0; ly < yCount; ++ly)
                for (int b = 0; b < 16; ++b) {
                    int lx, lz;
                    if (axis == 0) { lx = fixed; lz = b; }
                    else           { lx = b; lz = fixed; }
                    int ci = ColIdx(lx, ly, lz);
                    if ((s.bits[ci >> 6] >> (ci & 63)) & 1ull) continue;
                    if ((e.bits[ci >> 6] >> (ci & 63)) & 1ull) continue;
                    e.bits[ci >> 6] |= (uint64_t)1 << (ci & 63);
                    q.push_back({ cx, cz, lx, ly, lz });
                }
            }
        }
        printf("[skycull] seeded %zu cells, BFS...\n", q.size());

        // BFS propagate.
        size_t bfsVisited = 0;
        const int kNbr6[6][3] = {
            { +1, 0, 0 }, { -1, 0, 0 },
            { 0, +1, 0 }, { 0, -1, 0 },
            { 0, 0, +1 }, { 0, 0, -1 },
        };
        while (!q.empty()) {
            QE c = q.front(); q.pop_front();
            ++bfsVisited;
            for (int d = 0; d < 6; ++d) {
                int ncx = c.cx, ncz = c.cz;
                int nx = c.lx + kNbr6[d][0];
                int ny = c.ly + kNbr6[d][1];
                int nz = c.lz + kNbr6[d][2];
                if (ny < 0 || ny >= yCount) continue;
                NormCol(ncx, ncz, nx, nz);
                auto itS = globalSolid.find(McKey(ncx, ncz));
                if (itS == globalSolid.end()) continue;       // absent col → already classified
                int ci = ColIdx(nx, ny, nz);
                if ((itS->second.bits[ci >> 6] >> (ci & 63)) & 1ull) continue;  // solid
                ColSolid& en = globalExposed[McKey(ncx, ncz)];
                if (en.bits.empty()) en.bits.assign(u64PerCol, 0);
                if ((en.bits[ci >> 6] >> (ci & 63)) & 1ull) continue;            // already exposed
                en.bits[ci >> 6] |= (uint64_t)1 << (ci & 63);
                q.push_back({ ncx, ncz, nx, ny, nz });
            }
            if ((bfsVisited & 0xFFFFFF) == 0) {
                printf("[skycull] BFS visited=%zu queue=%zu\n", bfsVisited, q.size());
            }
        }
        printf("[skycull] BFS done visited=%zu\n", bfsVisited);

        // 3. Derive kept bitset = solid AND (any 6-neighbor air cell is exposed OR absent-air col).
        size_t keptTotal = 0;
        for (auto& kv : globalSolid) {
            int cx = (int32_t)(kv.first & 0xFFFFFFFFu);
            int cz = (int32_t)(kv.first >> 32);
            ColSolid& s = kv.second;
            ColSolid& k = globalKept[McKey(cx, cz)];
            k.bits.assign(u64PerCol, 0);
            for (int ly = 0; ly < yCount; ++ly)
            for (int lz = 0; lz < 16; ++lz)
            for (int lx = 0; lx < 16; ++lx) {
                int ci = ColIdx(lx, ly, lz);
                if (!((s.bits[ci >> 6] >> (ci & 63)) & 1ull)) continue;
                bool exp = false;
                for (int d = 0; d < 6; ++d) {
                    int ncx = cx, ncz = cz;
                    int nx = lx + kNbr6[d][0];
                    int ny = ly + kNbr6[d][1];
                    int nz = lz + kNbr6[d][2];
                    if (ny < 0)            { continue; }
                    if (ny >= yCount)      { exp = true; break; }     // above world = sky
                    NormCol(ncx, ncz, nx, nz);
                    auto itS = globalSolid.find(McKey(ncx, ncz));
                    if (itS == globalSolid.end()) {
                        if (AbsentColIsAir(ncx, ncz)) { exp = true; break; }
                        continue;                                      // absent-solid
                    }
                    int ni = ColIdx(nx, ny, nz);
                    if ((itS->second.bits[ni >> 6] >> (ni & 63)) & 1ull) continue;   // solid
                    auto itE = globalExposed.find(McKey(ncx, ncz));
                    if (itE == globalExposed.end()) continue;
                    if ((itE->second.bits[ni >> 6] >> (ni & 63)) & 1ull) { exp = true; break; }
                }
                if (exp) {
                    k.bits[ci >> 6] |= (uint64_t)1 << (ci & 63);
                    ++keptTotal;
                }
            }
        }
        auto Popcnt = [](uint64_t w) {
            w = w - ((w >> 1) & 0x5555555555555555ull);
            w = (w & 0x3333333333333333ull) + ((w >> 2) & 0x3333333333333333ull);
            w = (w + (w >> 4)) & 0x0F0F0F0F0F0F0F0Full;
            return (size_t)((w * 0x0101010101010101ull) >> 56);
        };
        size_t solidTotal = 0;
        for (auto& kv : globalSolid)
            for (uint64_t w : kv.second.bits) solidTotal += Popcnt(w);
        printf("[skycull] kept=%zu of %zu solid  (%.1f%% culled)\n",
               keptTotal, solidTotal, 100.0 - 100.0 * (double)keptTotal / std::max<size_t>(solidTotal, 1));

        // Free exposed bitset; not needed for emit.
        globalExposed.clear();
    }

    // ----- Height-map cull: keep surface + cliff/wall + N-voxel rind. -----
    if (heightCull) {
        // 1. Per-col heightmap: 16x16 int16 (=-1 if no solid in column).
        struct ColH { std::vector<int16_t> h; };  // size 256
        std::unordered_map<uint64_t, ColH> heights;
        heights.reserve(globalSolid.size() * 2);
        for (auto& kv : globalSolid) {
            int cx = (int32_t)(kv.first & 0xFFFFFFFFu);
            int cz = (int32_t)(kv.first >> 32);
            ColSolid& s = kv.second;
            ColH& h = heights[McKey(cx, cz)];
            h.h.assign(256, (int16_t)-1);
            // Scan top-down per (lx,lz).
            for (int lz = 0; lz < 16; ++lz)
            for (int lx = 0; lx < 16; ++lx) {
                int hi = -1;
                for (int ly = yCount - 1; ly >= 0; --ly) {
                    int ci = ColIdx(lx, ly, lz);
                    if ((s.bits[ci >> 6] >> (ci & 63)) & 1ull) { hi = ly; break; }
                }
                h.h[lz * 16 + lx] = (int16_t)hi;
            }
        }
        auto HAt = [&](int cx, int cz, int lx, int lz) -> int {
            // wrap lateral; cross-col lookup
            int dx = 0, dz = 0;
            if (lx < 0)        { dx = -1; lx += 16; }
            else if (lx >= 16) { dx = +1; lx -= 16; }
            if (lz < 0)        { dz = -1; lz += 16; }
            else if (lz >= 16) { dz = +1; lz -= 16; }
            auto it = heights.find(McKey(cx + dx, cz + dz));
            if (it == heights.end()) return -1;
            return it->second.h[lz * 16 + lx];
        };

        // 2. Seed bitset: surface voxels + cliff-face voxels.
        std::unordered_map<uint64_t, ColSolid> seedBits;
        seedBits.reserve(globalSolid.size() * 2);
        struct QE { int cx, cz, lx, ly, lz; };
        std::deque<QE> q;
        size_t seedCount = 0;
        for (auto& kv : globalSolid) {
            int cx = (int32_t)(kv.first & 0xFFFFFFFFu);
            int cz = (int32_t)(kv.first >> 32);
            ColSolid& s = kv.second;
            ColH& h = heights[McKey(cx, cz)];
            ColSolid& seed = seedBits[McKey(cx, cz)];
            seed.bits.assign(u64PerCol, 0);
            for (int lz = 0; lz < 16; ++lz)
            for (int lx = 0; lx < 16; ++lx) {
                int hi = h.h[lz * 16 + lx];
                if (hi < 0) continue;
                // Pre-fetch 4 lateral neighbour heights once per (lx,lz).
                int hpx = HAt(cx, cz, lx + 1, lz);
                int hnx = HAt(cx, cz, lx - 1, lz);
                int hpz = HAt(cx, cz, lx, lz + 1);
                int hnz = HAt(cx, cz, lx, lz - 1);
                for (int ly = hi; ly >= 0; --ly) {
                    int ci = ColIdx(lx, ly, lz);
                    if (!((s.bits[ci >> 6] >> (ci & 63)) & 1ull)) continue;
                    bool vis = false;
                    if (ly == hi) vis = true;
                    else if ((hpx >= 0 && hpx < ly) || (hnx >= 0 && hnx < ly)
                          || (hpz >= 0 && hpz < ly) || (hnz >= 0 && hnz < ly)) vis = true;
                    else if (hpx < 0 || hnx < 0 || hpz < 0 || hnz < 0) {
                        // neighbour col absent → treat as deep void → cliff face
                        vis = true;
                    }
                    if (vis) {
                        seed.bits[ci >> 6] |= (uint64_t)1 << (ci & 63);
                        q.push_back({ cx, cz, lx, ly, lz });
                        ++seedCount;
                    }
                }
            }
        }
        printf("[hcull] seeds=%zu (surface+cliffs), dilating by N=%d\n", seedCount, heightCullN);

        // 3. Dilate seed bits by N steps via 6-conn BFS inside solid cells.
        // Use level-by-level BFS.
        const int kNbr6[6][3] = {
            { +1, 0, 0 }, { -1, 0, 0 },
            { 0, +1, 0 }, { 0, -1, 0 },
            { 0, 0, +1 }, { 0, 0, -1 },
        };
        std::deque<QE> next;
        for (int step = 0; step < heightCullN; ++step) {
            while (!q.empty()) {
                QE c = q.front(); q.pop_front();
                for (int d = 0; d < 6; ++d) {
                    int ncx = c.cx, ncz = c.cz;
                    int nx = c.lx + kNbr6[d][0];
                    int ny = c.ly + kNbr6[d][1];
                    int nz = c.lz + kNbr6[d][2];
                    if (ny < 0 || ny >= yCount) continue;
                    NormCol(ncx, ncz, nx, nz);
                    auto itS = globalSolid.find(McKey(ncx, ncz));
                    if (itS == globalSolid.end()) continue;
                    int ni = ColIdx(nx, ny, nz);
                    if (!((itS->second.bits[ni >> 6] >> (ni & 63)) & 1ull)) continue;   // not solid
                    ColSolid& sn = seedBits[McKey(ncx, ncz)];
                    if (sn.bits.empty()) sn.bits.assign(u64PerCol, 0);
                    if ((sn.bits[ni >> 6] >> (ni & 63)) & 1ull) continue;               // already
                    sn.bits[ni >> 6] |= (uint64_t)1 << (ni & 63);
                    next.push_back({ ncx, ncz, nx, ny, nz });
                }
            }
            std::swap(q, next);
            printf("[hcull] dilate step=%d  frontier=%zu\n", step + 1, q.size());
        }

        // 4. Move seeds into globalKept; count.
        auto Popcnt = [](uint64_t w) {
            w = w - ((w >> 1) & 0x5555555555555555ull);
            w = (w & 0x3333333333333333ull) + ((w >> 2) & 0x3333333333333333ull);
            w = (w + (w >> 4)) & 0x0F0F0F0F0F0F0F0Full;
            return (size_t)((w * 0x0101010101010101ull) >> 56);
        };
        size_t keptTotal = 0, solidTotal = 0;
        for (auto& kv : seedBits)
            for (uint64_t w : kv.second.bits) keptTotal += Popcnt(w);
        for (auto& kv : globalSolid)
            for (uint64_t w : kv.second.bits) solidTotal += Popcnt(w);
        globalKept = std::move(seedBits);
        printf("[hcull] kept=%zu of %zu solid  (%.1f%% culled)\n",
               keptTotal, solidTotal,
               100.0 - 100.0 * (double)keptTotal / std::max<size_t>(solidTotal, 1));
    }

    // Either skyCull or heightCull populated globalKept; pass 2 uses KeptAt.
    bool kullActive = skyCull || heightCull;
    auto KeptAt = [&](int mcCX, int mcCZ, int lx, int ly, int lz) -> bool {
        if (ly < 0 || ly >= yCount) return false;
        NormCol(mcCX, mcCZ, lx, lz);
        auto it = globalKept.find(McKey(mcCX, mcCZ));
        if (it == globalKept.end()) return false;
        int ci = ColIdx(lx, ly, lz);
        return (it->second.bits[ci >> 6] >> (ci & 63)) & 1ull;
    };

    size_t totalVoxels = 0;
    size_t pass2Chunks = 0;
    ForEachChunk(regions, [&](int mcCX, int mcCZ, std::vector<SectionData>& secs, size_t ri) {
        int32_t chunkBaseX = mcCX * 16;
        int32_t chunkBaseZ = mcCZ * 16;
        for (auto& sd : secs) {
            int32_t baseY = (int32_t)sd.y * 16;
            if (baseY + 16 <= byMin || baseY >= byMax) continue;

            std::vector<int16_t> localToGlobal(sd.palette.size());
            bool anyKept = false;
            for (size_t pi = 0; pi < sd.palette.size(); ++pi) {
                localToGlobal[pi] = pal.Get(sd.palette[pi]);
                if (localToGlobal[pi] >= 0) anyKept = true;
            }
            if (!anyKept) continue;

            for (int y = 0; y < 16; ++y) {
                int32_t wy = baseY + y;
                if (wy < byMin || wy >= byMax) continue;
                int ly = wy - byMin;
                for (int z = 0; z < 16; ++z)
                for (int x = 0; x < 16; ++x) {
                    int li = (y * 16 + z) * 16 + x;
                    int pi = sd.indices[li];
                    if (pi < 0 || pi >= (int)localToGlobal.size()) continue;
                    int16_t g = localToGlobal[pi];
                    if (g < 0) continue;

                    // visMask + cull. When --sky-cull: voxel kept iff KeptAt set;
                    // visMask faces use kept neighbours (sky-occluded neighbour = solid).
                    uint8_t vm = 0;
                    if (kullActive) {
                        if (!KeptAt(mcCX, mcCZ, x, ly, z)) continue;
                        if (!KeptAt(mcCX, mcCZ, x + 1, ly, z)) vm |= 1u << 0;
                        if (!KeptAt(mcCX, mcCZ, x - 1, ly, z)) vm |= 1u << 1;
                        if (!KeptAt(mcCX, mcCZ, x, ly + 1, z)) vm |= 1u << 2;
                        if (!KeptAt(mcCX, mcCZ, x, ly, z + 1)) vm |= 1u << 4;
                        if (!KeptAt(mcCX, mcCZ, x, ly, z - 1)) vm |= 1u << 5;
                    } else {
                        if (!SolidAt(mcCX, mcCZ, x + 1, ly, z)) vm |= 1u << 0;
                        if (!SolidAt(mcCX, mcCZ, x - 1, ly, z)) vm |= 1u << 1;
                        if (!SolidAt(mcCX, mcCZ, x, ly + 1, z)) vm |= 1u << 2;
                        if (!SolidAt(mcCX, mcCZ, x, ly, z + 1)) vm |= 1u << 4;
                        if (!SolidAt(mcCX, mcCZ, x, ly, z - 1)) vm |= 1u << 5;
                        if (vm == 0) continue;          // fully occluded → cull
                    }

                    int32_t wx = chunkBaseX + x - bxMin;
                    int32_t wyo = wy - byMin;
                    int32_t wz = chunkBaseZ + z - bzMin;
                    int32_t cx = wx / D, lx = wx % D;
                    int32_t cy = wyo / D, lyc = wyo % D;
                    int32_t cz = wz / D, lz = wz % D;

                    DiskVoxel dv{};
                    dv.x = (uint8_t)lx;
                    dv.y = (uint8_t)lyc;
                    dv.z = (uint8_t)lz;
                    dv.visMask = vm;
                    dv.paletteIdx = (uint16_t)g;
                    OutChunk& oc = outChunks[ChunkKey(cx, cy, cz)];
                    oc.voxels.push_back(dv);
                    ++totalVoxels;
                }
            }
        }
        ++pass2Chunks;
        if ((ri & 7) == 0 && (pass2Chunks & 1023) == 0) {
            printf("[pass2] regions ~%zu/%zu  chunks=%zu  voxels=%zu\n",
                   ri + 1, regions.size(), pass2Chunks, totalVoxels);
        }
    });
    printf("[pass2] done  chunks=%zu  voxels=%zu\n", pass2Chunks, totalVoxels);

    if (outChunks.empty()) {
        fprintf(stderr, "no voxels emitted (palette empty?)\n");
        return 1;
    }

    // Build sequential output: deterministic chunk order by (cy,cz,cx).
    struct ChunkRec { int32_t cx, cy, cz; OutChunk* oc; };
    std::vector<ChunkRec> chunks;
    chunks.reserve(outChunks.size());
    for (auto& kv : outChunks) {
        uint64_t k = kv.first;
        int32_t cx = (int32_t)(k & 0xFFFFFull);
        int32_t cy = (int32_t)((k >> 20) & 0xFFFFFull);
        int32_t cz = (int32_t)((k >> 40) & 0xFFFFFull);
        chunks.push_back({ cx, cy, cz, &kv.second });
    }
    std::sort(chunks.begin(), chunks.end(),
              [](const ChunkRec& a, const ChunkRec& b) {
                  if (a.cy != b.cy) return a.cy < b.cy;
                  if (a.cz != b.cz) return a.cz < b.cz;
                  return a.cx < b.cx;
              });

    // Write .vox.
    FILE* fo = fopen(outPath, "wb");
    if (!fo) { fprintf(stderr, "open %s for write failed\n", outPath); return 1; }

    char magic[4] = { 'V', 'X', 'L', '3' };
    fwrite(magic, 1, 4, fo);
    uint32_t version = kAssetVersion;
    fwrite(&version, 4, 1, fo);
    uint32_t cd = (uint32_t)D;
    fwrite(&cd, 4, 1, fo);
    uint32_t cc = (uint32_t)chunks.size();
    fwrite(&cc, 4, 1, fo);
    uint32_t tv = (uint32_t)totalVoxels;
    fwrite(&tv, 4, 1, fo);
    int32_t origin[3] = { bxMin, byMin, bzMin };
    fwrite(origin, 4, 3, fo);
    float sunDir[3] = { -0.4f, -1.0f, -0.3f };
    {
        float L = std::sqrt(sunDir[0] * sunDir[0] + sunDir[1] * sunDir[1] + sunDir[2] * sunDir[2]);
        sunDir[0] /= L; sunDir[1] /= L; sunDir[2] /= L;
    }
    fwrite(sunDir, 4, 3, fo);
    uint32_t palCount = (uint32_t)pal.rgb.size();
    fwrite(&palCount, 4, 1, fo);
    for (uint32_t rgb : pal.rgb) {
        uint8_t r = (uint8_t)((rgb >> 16) & 0xFF);
        uint8_t g = (uint8_t)((rgb >> 8) & 0xFF);
        uint8_t b = (uint8_t)(rgb & 0xFF);
        uint32_t rgba = ((uint32_t)0xFFu << 24) | ((uint32_t)b << 16) | ((uint32_t)g << 8) | r;
        fwrite(&rgba, 4, 1, fo);
    }

    // Chunk metas (need offsets pre-compute).
    uint32_t off = 0;
    std::vector<uint32_t> offs(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
        offs[i] = off;
        off += (uint32_t)chunks[i].oc->voxels.size();
    }
    for (size_t i = 0; i < chunks.size(); ++i) {
        ChunkMeta m{};
        m.cx = (uint16_t)chunks[i].cx;
        m.cy = (uint16_t)chunks[i].cy;
        m.cz = (uint16_t)chunks[i].cz;
        m._pad = 0;
        m.voxelCount = (uint32_t)chunks[i].oc->voxels.size();
        m.voxelOffset = offs[i];
        fwrite(&m, sizeof(m), 1, fo);
    }
    // Voxel payload.
    for (auto& c : chunks) {
        if (!c.oc->voxels.empty())
            fwrite(c.oc->voxels.data(), sizeof(DiskVoxel), c.oc->voxels.size(), fo);
    }
    fclose(fo);

    printf("[mca2vox] wrote %s  chunks=%zu  voxels=%zu  palette=%zu\n",
           outPath, chunks.size(), totalVoxels, pal.rgb.size());
    if (!pal.unknown.empty()) {
        printf("[mca2vox] %zu unknown block names mapped via keyword fallback colour:\n",
               pal.unknown.size());
        for (const auto& n : pal.unknown) printf("    %s\n", n.c_str());
    }
    return 0;
}
