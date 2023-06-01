import sys
import os
import platform
import threading
import socket
import time
import ssl
import pytest
import contextlib
from Pyro5 import config, socketutil, protocol, errors, server, serializers, core
from Pyro5.svr_threads import SocketServer_Threadpool
from Pyro5.svr_multiplex import SocketServer_Multiplex


# determine ipv6 capability
has_ipv6 = socket.has_ipv6
if has_ipv6:
    s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    try:
        s.connect(("::1", 53))
        s.close()
        socket.getaddrinfo("localhost", 53, socket.AF_INET6)
    except socket.error:
        has_ipv6 = False


class TestSocketutil:
    @classmethod
    def setup_class(cls):
        config.POLLTIMEOUT = 0.1

    @pytest.mark.network
    def testGetIP(self):
        config.PREFER_IP_VERSION = 4
        myip = socketutil.get_ip_address("")
        assert len(str(myip)) > 4
        myip = socketutil.get_ip_address("", workaround127=True)
        assert len(str(myip)) > 4
        assert not str(myip).startswith("127.")
        addr = socketutil.get_ip_address("127.0.0.1", workaround127=False)
        assert "127.0.0.1" == str(addr)
        assert addr.version == 4
        addr = socketutil.get_ip_address("127.0.0.1", workaround127=True)
        assert "127.0.0.1" != str(addr)
        assert addr.version == 4

    def testGetIP6(self):
        if not has_ipv6:
            pytest.skip("no ipv6 capability")
        addr = socketutil.get_ip_address("::1", version=6)
        assert addr.version == 6
        assert ":" in str(addr)
        addr = socketutil.get_ip_address("localhost", version=6)
        assert addr.version == 6
        assert ":" in str(addr)

    def testGetInterface(self):
        addr = socketutil.get_interface("localhost")
        assert addr.version == 4
        assert str(addr).startswith("127.")
        assert str(addr.ip).startswith("127.0")
        assert str(addr.network).startswith("127.0")
        if has_ipv6:
            addr = socketutil.get_interface("::1")
            assert addr.version == 6
            assert ":" in str(addr)
            assert ":" in str(addr.ip)
            assert ":" in str(addr.network)

    def testUnusedPort(self):
        port1 = socketutil.find_probably_unused_port()
        port2 = socketutil.find_probably_unused_port()
        assert port1 > 0
        assert port1 != port2
        port1 = socketutil.find_probably_unused_port(socktype=socket.SOCK_DGRAM)
        port2 = socketutil.find_probably_unused_port(socktype=socket.SOCK_DGRAM)
        assert port1 > 0
        assert port1 != port2

    def testUnusedPort6(self):
        if not has_ipv6:
            pytest.skip("no ipv6 capability")
        port1 = socketutil.find_probably_unused_port(family=socket.AF_INET6)
        port2 = socketutil.find_probably_unused_port(family=socket.AF_INET6)
        assert port1 > 0
        assert port1 != port2
        port1 = socketutil.find_probably_unused_port(family=socket.AF_INET6, socktype=socket.SOCK_DGRAM)
        port2 = socketutil.find_probably_unused_port(family=socket.AF_INET6, socktype=socket.SOCK_DGRAM)
        assert port1 > 0
        assert port1 != port2

    def testBindUnusedPort(self):
        sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port1 = socketutil.bind_unused_port(sock1)
        port2 = socketutil.bind_unused_port(sock2)
        assert port1 > 0
        assert port1 != port2
        assert sock1.getsockname() == ("127.0.0.1", port1)
        sock1.close()
        sock2.close()

    def testBindUnusedPort6(self):
        if not has_ipv6:
            pytest.skip("no ipv6 capability")
        sock1 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock2 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        port1 = socketutil.bind_unused_port(sock1)
        port2 = socketutil.bind_unused_port(sock2)
        assert port1 > 0
        assert port1 != port2
        host, port, _, _ = sock1.getsockname()
        assert ":" in host
        assert port1 == port
        sock1.close()
        sock2.close()

    def testCreateUnboundSockets(self):
        s = socketutil.create_socket()
        assert socket.AF_INET == s.family
        bs = socketutil.create_bc_socket()
        assert socket.AF_INET == bs.family
        with contextlib.suppress(socket.error):
            host, port = s.getsockname()
            # can either fail with socket.error or return (host,0)
            assert 0 == port
        with contextlib.suppress(socket.error):
            host, port = bs.getsockname()
            # can either fail with socket.error or return (host,0)
            assert 0 == port
        s.close()
        bs.close()

    def testCreateUnboundSockets6(self):
        if not has_ipv6:
            pytest.skip("no ipv6 capability")
        s = socketutil.create_socket(ipv6=True)
        assert socket.AF_INET6 == s.family
        bs = socketutil.create_bc_socket(ipv6=True)
        assert socket.AF_INET6 == bs.family
        with contextlib.suppress(socket.error):
            host, port, _, _ = s.getsockname()
            # can either fail with socket.error or return (host,0)
            assert 0 == port
        with contextlib.suppress(socket.error):
            host, port, _, _ = bs.getsockname()
            # can either fail with socket.error or return (host,0)
            assert 0 == port
        s.close()
        bs.close()

    def testCreateBoundSockets(self):
        s = socketutil.create_socket(bind=('127.0.0.1', 0))
        assert socket.AF_INET == s.family
        bs = socketutil.create_bc_socket(bind=('127.0.0.1', 0))
        assert '127.0.0.1' == s.getsockname()[0]
        assert '127.0.0.1' == bs.getsockname()[0]
        s.close()
        bs.close()
        with pytest.raises(ValueError):
            socketutil.create_socket(bind=('localhost', 12345), connect=('localhost', 1234))

    def testCreateBoundSockets6(self):
        if not has_ipv6:
            pytest.skip("no ipv6 capability")
        s = socketutil.create_socket(bind=('::1', 0))
        assert socket.AF_INET6 == s.family
        bs = socketutil.create_bc_socket(bind=('::1', 0))
        assert ':' in s.getsockname()[0]
        assert ':' in bs.getsockname()[0]
        s.close()
        bs.close()
        with pytest.raises(ValueError):
            socketutil.create_socket(bind=('::1', 12345), connect=('::1', 1234))

    def testCreateBoundUnixSockets(self):
        if not hasattr(socket, "AF_UNIX"):
            pytest.skip("no unix domain sockets capability")
        SOCKNAME = "test_unixsocket"
        if os.path.exists(SOCKNAME):
            os.remove(SOCKNAME)
        s = socketutil.create_socket(bind=SOCKNAME)
        assert socket.AF_UNIX == s.family
        assert SOCKNAME == s.getsockname()
        s.close()
        if os.path.exists(SOCKNAME):
            os.remove(SOCKNAME)
        with pytest.raises(ValueError):
            socketutil.create_socket(bind=SOCKNAME, connect=SOCKNAME)

    def testAbstractNamespace(self):
        if not hasattr(socket, "AF_UNIX") and not sys.platform.startswith("linux"):
            pytest.skip("no unix domain sockets capability, and not Linux")
        SOCKNAME = "\0test_unixsocket_abstract_ns"  # mind the \0 at the start
        s = socketutil.create_socket(bind=SOCKNAME)
        assert bytes(SOCKNAME, "ascii") == s.getsockname()
        s.close()

    def testSend(self):
        ss = socketutil.create_socket(bind=("localhost", 0))
        port = ss.getsockname()[1]
        cs = socketutil.create_socket(connect=("localhost", port))
        socketutil.send_data(cs, b"foobar!" * 10)
        cs.shutdown(socket.SHUT_WR)
        a = ss.accept()
        data = socketutil.receive_data(a[0], 5)
        assert b"fooba" == data
        data = socketutil.receive_data(a[0], 5)
        assert b"r!foo" == data
        a[0].close()
        ss.close()
        cs.close()

    def testSendUnix(self):
        if not hasattr(socket, "AF_UNIX"):
            pytest.skip("no unix domain sockets capability")
        SOCKNAME = "test_unixsocket"
        if os.path.exists(SOCKNAME):
            os.remove(SOCKNAME)
        ss = socketutil.create_socket(bind=SOCKNAME)
        cs = socketutil.create_socket(connect=SOCKNAME)
        socketutil.send_data(cs, b"foobar!" * 10)
        cs.shutdown(socket.SHUT_WR)
        a = ss.accept()
        data = socketutil.receive_data(a[0], 5)
        assert b"fooba" == data
        data = socketutil.receive_data(a[0], 5)
        assert b"r!foo" == data
        a[0].close()
        ss.close()
        cs.close()
        if os.path.exists(SOCKNAME):
            os.remove(SOCKNAME)

    @pytest.mark.network
    def testBroadcast(self):
        ss = socketutil.create_bc_socket((None, 0))
        port = ss.getsockname()[1]
        cs = socketutil.create_bc_socket()
        for bcaddr in config.BROADCAST_ADDRS:
            try:
                cs.sendto(b"monkey", 0, (bcaddr, port))
            except socket.error as x:
                err = getattr(x, "errno", x.args[0])
                # handle some errno that some platforms like to throw
                if err not in socketutil.ERRNO_EADDRNOTAVAIL and err not in socketutil.ERRNO_EADDRINUSE:
                    raise
        data, _ = ss.recvfrom(500)
        assert b"monkey" == data
        cs.close()
        ss.close()

    def testMsgWaitallProblems(self):
        ss = socketutil.create_socket(bind=("localhost", 0), timeout=2)
        port = ss.getsockname()[1]
        cs = socketutil.create_socket(connect=("localhost", port), timeout=2)
        a = ss.accept()
        # test some sizes that might be problematic with MSG_WAITALL and check that they work fine
        for size in [1000, 10000, 32000, 32768, 32780, 41950, 41952, 42000, 65000, 65535, 65600, 80000]:
            socketutil.send_data(cs, b"x" * size)
            data = socketutil.receive_data(a[0], size)
            socketutil.send_data(a[0], data)
            data = socketutil.receive_data(cs, size)
            assert size == len(data)
        a[0].close()
        ss.close()
        cs.close()

    def testMsgWaitallProblems2(self):
        class ReceiveThread(threading.Thread):
            def __init__(self, sock, sizes):
                super(ReceiveThread, self).__init__()
                self.sock = sock
                self.sizes = sizes

            def run(self):
                cs, _ = self.sock.accept()
                for size in self.sizes:
                    data = socketutil.receive_data(cs, size)
                    socketutil.send_data(cs, data)
                cs.close()

        ss = socketutil.create_socket(bind=("localhost", 0))
        SIZES = [1000, 10000, 32000, 32768, 32780, 41950, 41952, 42000, 65000, 65535, 65600, 80000, 999999]
        serverthread = ReceiveThread(ss, SIZES)
        serverthread.daemon = True
        serverthread.start()
        port = ss.getsockname()[1]
        cs = socketutil.create_socket(connect=("localhost", port), timeout=2)
        # test some sizes that might be problematic with MSG_WAITALL and check that they work fine
        for size in SIZES:
            socketutil.send_data(cs, b"x" * size)
            data = socketutil.receive_data(cs, size)
            assert size == len(data)
        serverthread.join()
        ss.close()
        cs.close()

    def testMsgWaitAllConfig(self):
        if platform.system() == "Windows":
            # default config should be False on these platforms even though socket.MSG_WAITALL might exist
            assert not socketutil.USE_MSG_WAITALL
        else:
            # on all other platforms, default config should be True (as long as socket.MSG_WAITALL exists)
            if hasattr(socket, "MSG_WAITALL"):
                assert socketutil.USE_MSG_WAITALL
            else:
                assert not socketutil.USE_MSG_WAITALL



class ServerTestDaemon(server.Daemon):
    pass


class ServerCallback(object):
    def _handshake(self, connection, denied_reason=None):
        raise RuntimeError("this handshake method should never be called")

    def handleRequest(self, connection):
        if not isinstance(connection, socketutil.SocketConnection):
            raise TypeError("handleRequest expected SocketConnection parameter")
        msg = protocol.recv_stub(connection, [protocol.MSG_PING])
        if msg.type == protocol.MSG_PING:
            msg = protocol.SendingMessage(protocol.MSG_PING, 0, msg.seq, msg.serializer_id, b"ping")
            connection.send(msg.data)
        else:
            print("unhandled message type", msg.type)
            connection.close()

    def _housekeeping(self):
        pass


class ServerCallback_BrokenHandshake(ServerCallback):
    def _handshake(self, connection, denied_reason=None):
        raise ZeroDivisionError("handshake crashed (on purpose)")


class TestSocketServer:
    def testServer_thread(self):
        daemon = ServerCallback()
        port = socketutil.find_probably_unused_port()
        serv = SocketServer_Threadpool()
        serv.init(daemon, "localhost", port)
        assert serv.locationStr == "localhost:" + str(port)
        assert serv.sock is not None
        conn = socketutil.SocketConnection(serv.sock, "ID12345")
        assert conn.objectId == "ID12345"
        assert conn.sock is not None
        conn.close()
        conn.close()
        assert conn.sock is not None, "connections keep their socket object even if it's closed"
        serv.close()
        serv.close()
        assert serv.sock is None

    def testServer_multiplex(self):
        daemon = ServerCallback()
        port = socketutil.find_probably_unused_port()
        serv = SocketServer_Multiplex()
        serv.init(daemon, "localhost", port)
        assert serv.locationStr == "localhost:" + str(port)
        assert serv.sock is not None
        conn = socketutil.SocketConnection(serv.sock, "ID12345")
        assert conn.objectId == "ID12345"
        assert conn.sock is not None
        conn.close()
        conn.close()
        assert conn.sock is not None, "connections keep their socket object even if it's closed"
        serv.close()
        serv.close()
        assert serv.sock is None


class TestServerDOS_multiplex:
    def setup_method(self):
        self.orig_poll_timeout = config.POLLTIMEOUT
        self.orig_comm_timeout = config.COMMTIMEOUT
        config.POLLTIMEOUT = 0.5
        config.COMMTIMEOUT = 0.5
        self.socket_server = SocketServer_Multiplex

    def teardown_method(self):
        config.POLLTIMEOUT = self.orig_poll_timeout
        config.COMMTIMEOUT = self.orig_comm_timeout

    class ServerThread(threading.Thread):
        def __init__(self, server, daemon):
            threading.Thread.__init__(self)
            self.serv = server()
            self.serv.init(daemon(), "localhost", 0)
            self.locationStr = self.serv.locationStr
            self.stop_loop = threading.Event()

        def run(self):
            self.serv.loop(loopCondition=lambda: not self.stop_loop.is_set())
            self.serv.close()

    def testConnectCrash(self):
        serv_thread = TestServerDOS_multiplex.ServerThread(self.socket_server, ServerCallback_BrokenHandshake)
        serv_thread.start()
        time.sleep(0.2)
        assert serv_thread.is_alive(), "server thread failed to start"
        threadpool = getattr(serv_thread.serv, "pool", None)
        if threadpool:
            assert len(threadpool.idle) == 1
            assert len(threadpool.busy) == 0
        try:
            host, port = serv_thread.locationStr.split(':')
            port = int(port)
            try:
                # first connection attempt (will fail because server daemon _handshake crashes)
                csock = socketutil.create_socket(connect=(host, port))
                conn = socketutil.SocketConnection(csock, "uri")
                protocol.recv_stub(conn, [protocol.MSG_CONNECTOK])
            except errors.ConnectionClosedError:
                pass
            conn.close()
            time.sleep(0.1)
            if threadpool:
                assert len(threadpool.idle) == 1
                assert len(threadpool.busy) == 0
            try:
                # second connection attempt, should still work (i.e. server should still be running)
                csock = socketutil.create_socket(connect=(host, port))
                conn = socketutil.SocketConnection(csock, "uri")
                protocol.recv_stub(conn, [protocol.MSG_CONNECTOK])
            except errors.ConnectionClosedError:
                pass
        finally:
            if conn:
                conn.close()
            serv_thread.stop_loop.set()
            serv_thread.join()

    def testInvalidMessageCrash(self):
        serv_thread = TestServerDOS_multiplex.ServerThread(self.socket_server, ServerTestDaemon)
        serv_thread.start()
        time.sleep(0.2)
        assert serv_thread.is_alive(), "server thread failed to start"
        threadpool = getattr(serv_thread.serv, "pool", None)
        if threadpool:
            assert len(threadpool.idle) == 1
            assert len(threadpool.busy) == 0

        def connect(host, port):
            # connect to the server
            csock = socketutil.create_socket(connect=(host, port))
            conn = socketutil.SocketConnection(csock, "uri")
            # send the handshake/connect data
            ser = serializers.serializers_by_id[serializers.MarshalSerializer.serializer_id]
            data = ser.dumps({"handshake": "hello", "object": core.DAEMON_NAME})
            msg = protocol.SendingMessage(protocol.MSG_CONNECT, 0, 0, serializers.MarshalSerializer.serializer_id, data)
            conn.send(msg.data)
            # get the handshake/connect response
            protocol.recv_stub(conn, [protocol.MSG_CONNECTOK])
            return conn

        conn = None
        try:
            host, port = serv_thread.locationStr.split(':')
            port = int(port)
            conn = connect(host, port)
            # invoke something, but screw up the message (in this case, mess with the protocol version)
            orig_protocol_version = protocol.PROTOCOL_VERSION
            protocol.PROTOCOL_VERSION = 9999
            msgbytes = protocol.SendingMessage(protocol.MSG_PING, 42, 0, 0, b"something").data
            protocol.PROTOCOL_VERSION = orig_protocol_version
            conn.send(msgbytes)  # this should cause an error in the server because of invalid msg
            try:
                msg = protocol.recv_stub(conn, [protocol.MSG_RESULT])
                data = msg.data.decode("ascii", errors="ignore")  # convert raw message to string to check some stuff
                assert "Traceback" in data
                assert "ProtocolError" in data
                assert "version" in data
            except errors.ConnectionClosedError:
                # invalid message can cause the connection to be closed, this is fine
                pass
            # invoke something again, this should still work (server must still be running, but our client connection was terminated)
            conn.close()
            time.sleep(0.1)
            if threadpool:
                assert len(threadpool.idle) == 1
                assert len(threadpool.busy) == 0
            conn = connect(host, port)
            msg = protocol.SendingMessage(protocol.MSG_PING, 42, 999, 0, b"something")  # a valid message this time
            conn.send(msg.data)
            msg = protocol.recv_stub(conn, [protocol.MSG_PING])
            assert msg.type == protocol.MSG_PING
            assert msg.seq == 999
            assert msg.data == b"pong"
        finally:
            if conn:
                conn.close()
            serv_thread.stop_loop.set()
            serv_thread.join()


class TestServerDOS_threading(TestServerDOS_multiplex):
    def setup_method(self):
        super().setup_method()
        self.socket_server = SocketServer_Threadpool
        self.orig_numthreads = config.THREADPOOL_SIZE
        self.orig_numthreads_min = config.THREADPOOL_SIZE_MIN
        config.THREADPOOL_SIZE = 1
        config.THREADPOOL_SIZE_MIN = 1

    def teardown_method(self):
        config.THREADPOOL_SIZE = self.orig_numthreads
        config.THREADPOOL_SIZE_MIN = self.orig_numthreads_min


class TestSSL:
    def testContextAndSock(self):
        cert_dir = "../../certs"
        if not os.path.isdir(cert_dir):
            cert_dir = "../certs"
            if not os.path.isdir(cert_dir):
                cert_dir = "./certs"
                if not os.path.isdir(cert_dir):
                    raise IOError("cannot locate test certs directory")
        try:
            config.SSL = True
            config.SSL_REQUIRECLIENTCERT = True
            server_ctx = socketutil.get_ssl_context(cert_dir+"/server_cert.pem", cert_dir+"/server_key.pem")
            client_ctx = socketutil.get_ssl_context(clientcert=cert_dir+"/client_cert.pem", clientkey=cert_dir+"/client_key.pem")
            assert server_ctx.verify_mode == ssl.CERT_REQUIRED
            assert client_ctx.verify_mode == ssl.CERT_REQUIRED
            assert client_ctx.check_hostname
            sock = socketutil.create_socket(sslContext=server_ctx)
            try:
                assert hasattr(sock, "getpeercert")
            finally:
                sock.close()
        finally:
            config.SSL = False
