using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace IPCV.Bridge
{
    // Reads UDP packets on a background thread. Used for both tracking and faces.
    internal sealed class UdpInbox : IDisposable
    {
        internal sealed class Packet { public byte[] bytes; public double receiptSeconds; }
        private readonly UdpClient client;
        private readonly Thread thread;
        private readonly Queue<Packet> queue = new Queue<Packet>();
        private readonly int maxBytes, capacity;
        private volatile bool stopping;
        private long received, drops;
        private string error;
        public long ReceivedPackets => Interlocked.Read(ref received);
        public long Drops => Interlocked.Read(ref drops);
        public string Error => Volatile.Read(ref error);
        public static double Now() => BridgeClock.Now();

        public UdpInbox(int port, int maxBytes, int capacity)
        {
            if (port < 1 || port > 65535) throw new ArgumentOutOfRangeException("port");
            this.maxBytes = maxBytes;
            this.capacity = capacity;
            client = new UdpClient(AddressFamily.InterNetwork);
            try
            {
                client.ExclusiveAddressUse = true;
                client.Client.Bind(new IPEndPoint(IPAddress.Loopback, port));
                client.Client.ReceiveTimeout = 100;
                thread = new Thread(Receive) { IsBackground = true, Name = "IPCV UDP " + port };
                thread.Start();
            }
            catch { client.Close(); throw; }
        }

        private void Receive()
        {
            var source = new IPEndPoint(IPAddress.Any, 0);
            while (!stopping)
            {
                try
                {
                    byte[] bytes = client.Receive(ref source);
                    Interlocked.Increment(ref received);
                    if (bytes.Length == 0 || bytes.Length > maxBytes) { Interlocked.Increment(ref drops); continue; }
                    var packet = new Packet { bytes = bytes, receiptSeconds = Now() };
                    lock (queue)
                    {
                        if (queue.Count == capacity) { queue.Dequeue(); Interlocked.Increment(ref drops); }
                        queue.Enqueue(packet);
                    }
                }
                catch (SocketException e)
                {
                    if (e.SocketErrorCode == SocketError.TimedOut) continue;
                    if (!stopping) Interlocked.Exchange(ref error, e.Message);
                    return;
                }
                catch (ObjectDisposedException) { return; }
            }
        }

        public Packet[] Drain()
        {
            lock (queue)
            {
                Packet[] packets = queue.ToArray();
                queue.Clear();
                return packets;
            }
        }

        public void Dispose() { stopping = true; client.Close(); thread.Join(500); }
    }
}
