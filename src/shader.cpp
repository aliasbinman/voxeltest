#include "shader.h"

#include <atomic>
#include <vector>
#include <windows.h>
#if !defined(SHADER_XBOX)
  #include <filesystem>
  #include <fstream>
#endif

using Microsoft::WRL::ComPtr;

#if !defined(VOXELTEST_XBOX)
bool ShaderCompiler::Init()
{
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils_)))) return false;
    if (FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler_)))) return false;
    if (FAILED(utils_->CreateDefaultIncludeHandler(&include_))) return false;
    return true;
}
#else
bool ShaderCompiler::Init() { return true; /* Xbox loads pre-built .cso via LoadCso */ }
#endif

#if !defined(VOXELTEST_XBOX)
bool ShaderCompiler::Compile(const std::wstring& path,
                             const wchar_t* entry,
                             const wchar_t* profile,
                             const std::vector<std::wstring>& defines,
                             ComPtr<IDxcBlob>& outBlob,
                             std::string* outError)
{
    if (!compiler_) { if (outError) *outError = "ShaderCompiler not initialized"; return false; }

    // Load file from disk into a blob.
    ComPtr<IDxcBlobEncoding> source;
    if (FAILED(utils_->LoadFile(path.c_str(), nullptr, &source)))
    {
        if (outError) *outError = "LoadFile failed";
        return false;
    }

    DxcBuffer buf{};
    buf.Ptr      = source->GetBufferPointer();
    buf.Size     = source->GetBufferSize();
    buf.Encoding = DXC_CP_ACP;

    // Build arg list. Note: pcwszArgs in Compile() takes LPCWSTR[].
    std::vector<std::wstring> args;
    args.push_back(path);
    args.push_back(L"-E"); args.push_back(entry);
    args.push_back(L"-T"); args.push_back(profile);
    args.push_back(L"-HV"); args.push_back(L"2021");
#ifdef _DEBUG
    args.push_back(L"-Zi");
    args.push_back(L"-Qembed_debug");
#else
    args.push_back(L"-O3");
#endif

    // Resolve includes relative to the shader's directory.
    std::filesystem::path p(path);
    std::wstring incDir = p.parent_path().wstring();
    if (!incDir.empty())
    {
        args.push_back(L"-I");
        args.push_back(incDir);
    }

    for (const auto& d : defines)
    {
        args.push_back(L"-D");
        args.push_back(d);
    }

    std::vector<LPCWSTR> argv;
    argv.reserve(args.size());
    for (const auto& s : args) argv.push_back(s.c_str());

    ComPtr<IDxcResult> result;
    HRESULT hr = compiler_->Compile(&buf, argv.data(), (UINT32)argv.size(),
                                    include_.Get(), IID_PPV_ARGS(&result));
    if (FAILED(hr))
    {
        if (outError) *outError = "IDxcCompiler3::Compile call failed";
        return false;
    }

    // Surface compile errors before checking status.
    ComPtr<IDxcBlobUtf8> errors;
    if (SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr))
        && errors && errors->GetStringLength() > 0)
    {
        if (outError) *outError = errors->GetStringPointer();
    }

    HRESULT status = S_OK;
    result->GetStatus(&status);
    if (FAILED(status)) return false;

    if (FAILED(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&outBlob), nullptr)))
    {
        if (outError) *outError = "GetOutput(DXC_OUT_OBJECT) failed";
        return false;
    }
    return true;
}
#else
bool ShaderCompiler::Compile(const std::wstring&, const wchar_t*, const wchar_t*,
                             const std::vector<std::wstring>&, ComPtr<IDxcBlob>&,
                             std::string* outError)
{
    if (outError) *outError = "Compile() unavailable on Xbox — use LoadCso() instead";
    return false;
}
#endif

// Minimal IDxcBlob implementation that owns a heap-allocated byte buffer.
// Used by LoadCso so we can avoid calling DxcCreateInstance on Xbox (no DXC
// runtime there). Implements just IUnknown + GetBufferPointer/Size.
namespace {
    struct OwnedBlob final : public IDxcBlob
    {
        std::vector<uint8_t> data;
        std::atomic<ULONG>   refs{1};

        // IUnknown
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override
        {
            if (!ppv) return E_POINTER;
            if (riid == __uuidof(IUnknown) || riid == __uuidof(IDxcBlob)) {
                *ppv = static_cast<IDxcBlob*>(this);
                AddRef();
                return S_OK;
            }
            *ppv = nullptr;
            return E_NOINTERFACE;
        }
        ULONG STDMETHODCALLTYPE AddRef()  override { return ++refs; }
        ULONG STDMETHODCALLTYPE Release() override
        {
            ULONG n = --refs;
            if (n == 0) delete this;
            return n;
        }
        // IDxcBlob
        LPVOID STDMETHODCALLTYPE GetBufferPointer() override { return data.data(); }
        SIZE_T STDMETHODCALLTYPE GetBufferSize()    override { return data.size(); }
    };
}

// LoadCso — read a pre-built .cso into an IDxcBlob without invoking DXC. Lets
// Xbox load shaders at runtime where dxcompiler.dll isn't available.
bool ShaderCompiler::LoadCso(const std::wstring& csoPath,
                             ComPtr<IDxcBlob>& outBlob,
                             std::string* outError)
{
    HANDLE f = CreateFileW(csoPath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE)
    {
        if (outError) {
            std::string narrow; narrow.reserve(csoPath.size());
            for (wchar_t w : csoPath) narrow.push_back(w < 128 ? (char)w : '?');
            *outError = "CreateFileW failed for " + narrow;
        }
        return false;
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f, &sz)) { CloseHandle(f); return false; }
    auto blob = new OwnedBlob();
    blob->data.resize((size_t)sz.QuadPart);
    DWORD got = 0;
    if (!ReadFile(f, blob->data.data(), (DWORD)sz.QuadPart, &got, nullptr) || got != sz.QuadPart)
    {
        CloseHandle(f);
        blob->Release();
        if (outError) *outError = "ReadFile failed/short for cso";
        return false;
    }
    CloseHandle(f);
    outBlob.Attach(blob); // takes ref from constructor
    return true;
}
