#pragma once
// Minimal DXC compile wrapper. On PC: runtime IDxcCompiler3.Compile from .hlsl.
// On Xbox: pre-built .cso loaded via raw file IO (no DXC runtime). The class
// keeps the same public surface on both platforms; Init() + Compile() are
// stubbed out on Xbox where dxcapi.h / d3d12.h can't be included (they conflict
// with d3d12_xs.h).

#if defined(_GAMING_XBOX_SCARLETT) || defined(_GAMING_XBOX_XBOXONE)
  #define SHADER_XBOX 1
#endif

#if !defined(SHADER_XBOX)
  #include <d3d12.h>
  #include <dxcapi.h>
#else
  // Minimal IDxcBlob declaration. Xbox can't include <dxcapi.h> (drags in
  // stock <d3d12.h> which fights d3d12_xs.h). We only need the IDxcBlob
  // interface to pass shader bytecode pointer + size around. IID matches the
  // real IDxcBlob so the type is binary-compatible if anything later does
  // touch it. OwnedBlob in shader.cpp implements this interface.
  #include <unknwn.h>
  struct __declspec(uuid("8BA5FB08-5195-40e2-AC58-0D989C3A0102")) IDxcBlob : public IUnknown
  {
      virtual LPVOID STDMETHODCALLTYPE GetBufferPointer() = 0;
      virtual SIZE_T STDMETHODCALLTYPE GetBufferSize() = 0;
  };
#endif

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
    bool LoadCso(const std::wstring& csoPath,
                 Microsoft::WRL::ComPtr<IDxcBlob>& outBlob,
                 std::string* outError = nullptr);

#if !defined(SHADER_XBOX)
private:
    Microsoft::WRL::ComPtr<IDxcUtils>          utils_;
    Microsoft::WRL::ComPtr<IDxcCompiler3>      compiler_;
    Microsoft::WRL::ComPtr<IDxcIncludeHandler> include_;
#endif
};
