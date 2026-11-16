# Prints every adapter DXGI enumerates, with the fields the Direct3D 12 backend uses to
# choose one: vendor and device IDs, DXGI_ADAPTER_FLAG bits, and whether
# D3D12CreateDevice accepts the adapter at feature level 12.0.
$ErrorActionPreference = 'Stop'

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class DxgiAdapterList
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct AdapterDesc1
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string Description;
        public uint VendorId;
        public uint DeviceId;
        public uint SubSysId;
        public uint Revision;
        public UIntPtr DedicatedVideoMemory;
        public UIntPtr DedicatedSystemMemory;
        public UIntPtr SharedSystemMemory;
        public uint LuidLowPart;
        public int LuidHighPart;
        public uint Flags;
    }

    [ComImport, Guid("770aae78-f26f-4dba-a829-253c83d1b387"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDXGIFactory1
    {
        void SetPrivateData();
        void SetPrivateDataInterface();
        void GetPrivateData();
        void GetParent();
        void EnumAdapters();
        void MakeWindowAssociation();
        void GetWindowAssociation();
        void CreateSwapChain();
        void CreateSoftwareAdapter();
        [PreserveSig] int EnumAdapters1(uint index, out IntPtr adapter);
    }

    [ComImport, Guid("29038f61-3839-4626-91fd-086879011a05"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDXGIAdapter1
    {
        void SetPrivateData();
        void SetPrivateDataInterface();
        void GetPrivateData();
        void GetParent();
        void EnumOutputs();
        void GetDesc();
        void CheckInterfaceSupport();
        [PreserveSig] int GetDesc1(out AdapterDesc1 desc);
    }

    [DllImport("dxgi.dll")]
    private static extern int CreateDXGIFactory1(ref Guid riid, [MarshalAs(UnmanagedType.Interface)] out IDXGIFactory1 factory);

    [DllImport("d3d12.dll")]
    private static extern int D3D12CreateDevice(IntPtr adapter, int minimumFeatureLevel, ref Guid riid, IntPtr device);

    public static string[] Describe()
    {
        var factoryId = new Guid("770aae78-f26f-4dba-a829-253c83d1b387");
        IDXGIFactory1 factory;
        Marshal.ThrowExceptionForHR(CreateDXGIFactory1(ref factoryId, out factory));
        var deviceId = new Guid("189819f1-1db6-4b57-be54-1821339b85f7");
        var lines = new System.Collections.Generic.List<string>();
        for (uint index = 0; ; ++index)
        {
            IntPtr pointer;
            if (factory.EnumAdapters1(index, out pointer) != 0)
            {
                break;
            }
            try
            {
                var adapter = (IDXGIAdapter1)Marshal.GetObjectForIUnknown(pointer);
                AdapterDesc1 desc;
                Marshal.ThrowExceptionForHR(adapter.GetDesc1(out desc));
                int probe = D3D12CreateDevice(pointer, 0xc000, ref deviceId, IntPtr.Zero);
                lines.Add(string.Format(
                    "{0}: {1} | vendor 0x{2:x4} device 0x{3:x4} subsys 0x{4:x8} rev {5} | flags 0x{6:x} | dedicated video {7} MiB | D3D12 FL 12.0 probe 0x{8:x8}",
                    index, desc.Description, desc.VendorId, desc.DeviceId, desc.SubSysId, desc.Revision,
                    desc.Flags, desc.DedicatedVideoMemory.ToUInt64() / (1024 * 1024), probe));
            }
            finally
            {
                Marshal.Release(pointer);
            }
        }
        return lines.ToArray();
    }
}
'@

[DxgiAdapterList]::Describe()
'Flags: 0x1 remote, 0x2 software. A probe of 0x00000001 (S_FALSE) means D3D12 accepts the adapter.'
