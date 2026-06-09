#include "shader.h"

#include <filesystem>
#include <fstream>

using Microsoft::WRL::ComPtr;

bool ShaderCompiler::Init()
{
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils_)))) return false;
    if (FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler_)))) return false;
    if (FAILED(utils_->CreateDefaultIncludeHandler(&include_))) return false;
    return true;
}

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
