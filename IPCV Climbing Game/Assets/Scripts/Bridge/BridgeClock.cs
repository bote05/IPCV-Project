using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace IPCV.Bridge
{
    public static class BridgeClock
    {
        private static readonly bool Windows = Environment.OSVersion.Platform == PlatformID.Win32NT;
        private static readonly double Frequency = Windows ? WindowsFrequency() : Stopwatch.Frequency;

        [DllImport("kernel32.dll")]
        private static extern bool QueryPerformanceCounter(out long counter);
        [DllImport("kernel32.dll")]
        private static extern bool QueryPerformanceFrequency(out long frequency);

        private static double WindowsFrequency()
        {
            QueryPerformanceFrequency(out long frequency);
            return frequency;
        }

        // Unity's Stopwatch starts from 0 when Unity starts, so it can't be compared
        // with Python. Reading QPC directly gives the same clock as perf_counter().
        public static double Now()
        {
            if (!Windows) return Stopwatch.GetTimestamp() / Frequency;
            QueryPerformanceCounter(out long counter);
            return counter / Frequency;
        }
    }
}
