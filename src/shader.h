#pragma once
// Minimal DXC-based HLSL compile wrapper. Runtime compile via IDxcCompiler3 +
// IDxcUtils. Include handler defaults to the file's own directory so
// `#include "shading.hlsli"` resolves the same as the DX11 D3DCompileFromFile
// path used to. SM 6.0 by default — adjust per call when we need 6.6+ atomics.
#include <d3d12.h>
#include <dxcapi.h>
#include <wrl/client.h>
#include <string>
#include <vector>

class ShaderCompiler
{
public:
    bool Init();
    bool Compile(const std::wstring& path,
                 const wchar_t* entry,
                 const wchar_t* profile,
                 const std::vector<std::wstring>& defines,
                 Microsoft::WRL::ComPtr<IDxcBlob>& outBlob,
                 std::string* outError = nullptr);

private:
    Microsoft::WRL::ComPtr<IDxcUtils>          utils_;
    Microsoft::WRL::ComPtr<IDxcCompiler3>      compiler_;
    Microsoft::WRL::ComPtr<IDxcIncludeHandler> include_;
};
