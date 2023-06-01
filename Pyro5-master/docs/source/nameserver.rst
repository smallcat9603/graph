.. index:: Name Server

.. _name-server:

***********
Name Server
***********

The Pyro Name Server is a tool to help keeping track of your objects in your network.
It is also a means to give your Pyro objects logical names instead of the need to always
know the exact object name (or id) and its location.

Pyro will name its objects like this::

    PYRO:obj_dcf713ac20ce4fb2a6e72acaeba57dfd@localhost:51850
    PYRO:custom_name@localhost:51851

It's either a generated unique object id on a certain host, or a name you chose yourself.
But to connect to these objects you'll always need to know the exact object name or id and
the exact hostname and port number of the Pyro daemon where the object is running.
This can get tedious, and if you move servers around (or Pyro objects) your client programs
can no longer connect to them until you update all URIs.

Enter the *name server*.
This is a simple phone-book like registry that maps logical object names to their corresponding URIs.
No need to remember the exact URI anymore. Instead, you can ask the name server to look it up for
you. You only need to give it the logical object name.

.. note:: Usually you only need to run *one single instance* of the name server in your network.
    You can start multiple name servers but they are unconnected; you'll end up with a partitioned name space.


**Example scenario:**
Assume you've got a document archive server that publishes a Pyro object with several archival related methods in it.
This archive server can register this object with the name server, using a logical name such as
"Department.ArchiveServer". Any client can now connect to it using only the name "Department.ArchiveServer".
They don't need to know the exact Pyro id and don't even need to know the location.
This means you can move the archive server to another machine and as long as it updates its record in the
name server, all clients won't notice anything and can keep on running without modification.


.. index:: starting the name server
    double: name server; command line

.. _nameserver-nameserver:

Starting the Name Server
========================

The easiest way to start a name server is by using the command line tool.

synopsys: :command:`python -m Pyro5.nameserver [options]` (or simply: :command:`pyro5-ns [options]`)


Starts the Pyro Name Server. It can run without any arguments but there are several that you
can use, for instance to control the hostname and port that the server is listening on.
A short explanation of the available options can be printed with the help option.
When it starts, it prints a message similar to this ('neptune' is the hostname of the machine it is running on)::

    $ pyro5-ns -n neptune
    Broadcast server running on 0.0.0.0:9091
    NS running on neptune:9090 (192.168.178.20)
    URI = PYRO:Pyro.NameServer@neptune:9090


As you can see it prints that it started a broadcast server (and its location),
a name server (and its location), and it also printed the URI that clients can use
to access it directly.

The nameserver uses a fast but volatile in-memory database by default. With a command line argument
you can select a persistent storage mechanism (see below). If you're using that, your registrations
will not be lost when the nameserver stops/restarts. The server will print the number of
existing registrations at startup time if it discovers any.


.. note::
    Pyro by default binds its servers on localhost which means you cannot reach them
    from another machine on the network. This behavior also applies to the name server.
    If you want to be able to talk to the name server from other machines, you have to
    explicitly provide a hostname or non-loopback interface to bind on.

There are several command line options for this tool:

.. program:: Pyro5.nameserver

.. option:: -h, --help

   Print a short help message and exit.

.. option:: -n HOST, --host=HOST

   Specify hostname or ip address to bind the server on.
   The default is localhost, note that your name server will then not be visible from the network
   If the server binds on localhost, *no broadcast responder* is started either.
   Make sure to provide a hostname or ip address to make the name server reachable from other machines, if you want that.

.. option:: -p PORT, --port=PORT

   Specify port to bind server on (0=random).

.. option:: -u UNIXSOCKET, --unixsocket=UNIXSOCKET

   Specify a Unix domain socket name to bind server on, rather than a normal TCP/IP socket.

.. option:: --bchost=BCHOST

   Specify the hostname or ip address to bind the broadcast responder on.
   Note: if the hostname where the name server binds on is localhost (or 127.0.x.x),
   no broadcast responder is started.

.. option:: --bcport=BCPORT

   Specify the port to bind the broadcast responder on (0=random).

.. option:: --nathost=NATHOST

   Specify the external host name to use in case of NAT

.. option:: --natport=NATPORT

   Specify the external port use in case of NAT

.. option:: -s STORAGE, --storage=STORAGE

   Specify the storage mechanism to use. You have several options:

    - ``memory`` - fast, volatile in-memory database. This is the default.
    - ``dbm:dbfile`` - dbm-style persistent database table. Provide the filename to use. This storage type does not support metadata.
    - ``sql:sqlfile`` - sqlite persistent database. Provide the filename to use.

.. option:: -x, --nobc

   Don't start a broadcast responder. Clients will not be able to use the UDP-broadcast lookup
   to discover this name server.
   (The broadcast responder listens to UDP broadcast packets on the local network subnet,
   to signal its location to clients that want to talk to the name server)


Starting the Name Server from within your own code
==================================================

Another way to start up a name server is by doing it from within your own code.
This is more complex than simply launching it via the command line tool,
because you have to integrate the name server into the rest of your program (perhaps you need to merge event loops?).
For your convenience, two helper functions are available to create a name server yourself:
:py:func:`Pyro5.nameserver.start_ns` and :py:func:`Pyro5.nameserver.start_ns_loop`.
Look at the `eventloop example <https://github.com/irmen/Pyro5/tree/master/examples/eventloop>`_ to see how you can use this.

**Custom storage mechanism:**
The utility functions allow you to specify a custom storage mechanism (via the ``storage`` parameter).
By default the in memory storage :py:class:`Pyro5.nameserver.MemoryStorage` is used.
In the :py:mod:`Pyro5.nameserver` module you can find the other implementation (sqlite).
You could also build your own, as long as it has the same interface.



.. index::
    double: name server; configuration items

Configuration items
===================
There are a couple of config items related to the nameserver.
They are used both by the name server itself (to configure the values it will use to start
the server with), and the client code that locates the name server (to give it optional hints where
the name server is located). Often these can be overridden with a command line option or with a method parameter in your code.

================== ===========
Configuration item description
================== ===========
HOST               hostname that the name server will bind on (being a regular Pyro daemon).
NS_HOST            the hostname or ip address of the name server. Used for locating in clients only.
NS_PORT            the port number of the name server. Used by the server and for locating in clients.
NS_BCHOST          the hostname or ip address of the name server's broadcast responder. Used only by the server.
NS_BCPORT          the port number of the name server's broadcast responder. Used by the server and for locating in clients.
NATHOST            the external hostname in case of NAT. Used only by the server.
NATPORT            the external port in case of NAT. Used only by the server.
NS_AUTOCLEAN       a recurring period in seconds where the Name server checks its registrations, and removes the ones that are no longer available. Defaults to 0.0 (off).
================== ===========


.. index::
    double: name server; name server control

.. _nameserver-nsc:

Name server control tool
========================
The name server control tool (or 'nsc') is used to talk to a running name server and perform
diagnostic or maintenance actions such as querying the registered objects, adding or removing
a name registration manually, etc.

synopsis: :command:`python -m Pyro5.nsc [options] command [arguments]` (or simply: :command:`pyro5-nsc [options] command [arguments]`)


.. program:: Pyro5.nsc

.. option:: -h, --help

   Print a short help message and exit.

.. option:: -n HOST, --host=HOST

   Provide the hostname or ip address of the name server.
   The default is to do a broadcast lookup to search for a name server.

.. option:: -p PORT, --port=PORT

   Provide the port of the name server, or its broadcast port if you're doing a broadcast lookup.

.. option:: -u UNIXSOCKET, --unixsocket=UNIXSOCKET

   Provide the Unix domain socket name of the name server, rather than a normal TCP/IP socket.

.. option:: -v, --verbose

   Print more output that could be useful.


The available commands for this tool are:

list : list [prefix]
  List all objects with their metadata registered in the name server. If you supply a prefix,
  the list will be filtered to show only the objects whose name starts with the prefix.

listmatching : listmatching pattern
  List only the objects with a name matching the given regular expression pattern.

lookup : lookup name
  Looks up a single name registration and prints the uri.

yplookup_all : yplookup_all metadata [metadata...]
  List the objects having *all* of the given metadata tags

yplookup_any : yplookup_any metadata [metadata...]
  List the objects having *any one* (or multiple) of the given metadata tags

register : register name uri
  Registers a name to the given Pyro object :abbr:`URI (universal resource identifier)`.

remove : remove name
  Removes the entry with the exact given name from the name server.

removematching : removematching pattern
  Removes all entries matching the given regular expression pattern.

setmeta : setmeta name [metadata...]
  Sets the new list of metadata tags for the given Pyro object.
  If you don't specify any metadata tags, the metadata of the object is cleared.

ping
  Does nothing besides checking if the name server is running and reachable.


Example::

  $ pyro5-nsc ping
  Name server ping ok.

  $ pyro5-nsc list Pyro
  --------START LIST - prefix 'Pyro'
  Pyro.NameServer --> PYRO:Pyro.NameServer@localhost:9090
      metadata: {'class:Pyro5.nameserver.NameServer'}
  --------END LIST - prefix 'Pyro'


.. index::
    double: name server; locating the name server

Locating the Name Server and using it in your code
==================================================
The name server is a Pyro object itself, and you access it through a normal Pyro proxy.
The object exposed is :class:`Pyro5.nameserver.NameServer`.
Getting a proxy for the name server is done using the following function:
:func:`Pyro5.core.locate_ns` (also available as :func:`Pyro5.api.locate_ns`).

.. index::
    double: name server; broadcast lookup

By far the easiest way to locate the Pyro name server is by using the broadcast lookup mechanism.
This goes like this: you simply ask Pyro to look up the name server and return a proxy for it.
It automatically figures out where in your subnet it is running by doing a broadcast and returning
the first Pyro name server that responds. The broadcast is a simple UDP-network broadcast, so this
means it usually won't travel outside your network subnet (or through routers) and your firewall
needs to allow UDP network traffic.

There is a config item ``BROADCAST_ADDRS`` that contains a comma separated list of the broadcast
addresses Pyro should use when doing a broadcast lookup. Depending on your network configuration,
you may have to change this list to make the lookup work. It could be that you have to add the
network broadcast address for the specific network that the name server is located on.

.. note::
    You can only talk to a name server on a different machine if it didn't bind on localhost (that
    means you have to start it with an explicit host to bind on). The broadcast lookup mechanism
    only works in this case as well -- it doesn't work with a name server that binds on localhost.
    For instance, the name server started as an example in :ref:`nameserver-nameserver` was told to
    bind on the host name 'neptune' and it started a broadcast responder as well.
    If you use the default host (localhost) a broadcast responder will not be created.

Normally, all name server lookups are done this way. In code, it is simply calling the
locator function without any arguments.
If you want to circumvent the broadcast lookup (because you know the location of the
server already, somehow) you can specify the hostname.
As soon as you provide a specific hostname to the name server locator (by using a host argument
to the ``locate_ns`` call, or by setting the ``NS_HOST`` config item, etc) it will no longer use
a broadcast too try to find the name server.

.. function:: locate_ns([host=None, port=None, broadcast=True])

    Get a proxy for a name server somewhere in the network.
    If you're not providing host or port arguments, the configured defaults are used.
    Unless you specify a host, a broadcast lookup is done to search for a name server.
    (api reference: :py:func:`Pyro5.core.locate_ns`)

    :param host: the hostname or ip address where the name server is running.
        Default is ``None`` which means it uses a network broadcast lookup.
        If you specify a host, no broadcast lookup is performed.
    :param port: the port number on which the name server is running.
        Default is ``None`` which means use the configured default.
        The exact meaning depends on whether the host parameter is given:

        * host parameter given: the port now means the actual name server port.
        * host parameter not given: the port now means the broadcast port.
    :param broadcast: should a broadcast be used to locate the name server, if
        no location is specified? Default is True.


.. index:: PYRONAME protocol type
.. _nameserver-pyroname:

The PYRONAME protocol type
==========================
To create a proxy and connect to a Pyro object, Pyro needs an URI so it can find the object.
Because it is so convenient, the name server logic has been integrated into Pyro's URI mechanism
by means of the special ``PYRONAME`` protocol type (rather than the normal ``PYRO`` protocol type).
This protocol type tells Pyro to treat the URI as a logical object name instead, and Pyro will
do a name server lookup automatically to get the actual object's URI. The form of a PYRONAME uri
is very simple::

    PYRONAME:some_logical_object_name
    PYRONAME:some_logical_object_name@nshostname           # with optional host name
    PYRONAME:some_logical_object_name@nshostname:nsport    # with optional host name + port

where "some_logical_object_name" is the name of a registered Pyro object in the name server.
When you also provide the ``nshostname`` and perhaps even ``nsport`` parts in the uri, you tell Pyro to look
for the name server on that specific location (instead of relying on a broadcast lookup mechanism).
(You can achieve more or less the same by setting the ``NS_HOST`` and ``NS_PORT`` config items)

All this means that instead of manually resolving objects like this::

    nameserver=Pyro5.core.locate_ns()
    uri=nameserver.lookup("Department.BackupServer")
    proxy=Pyro5.client.Proxy(uri)
    proxy.backup()

you can write this instead::

    proxy=Pyro5.client.Proxy("PYRONAME:Department.BackupServer")
    proxy.backup()

An additional benefit of using a PYRONAME uri in a proxy is that the proxy isn't strictly
tied to a specific object on a specific location. This is useful in scenarios where the server
objects might move to another location, for instance when a disconnect/reconnect occurs.
See the `autoreconnect example <https://github.com/irmen/Pyro5/tree/master/examples/autoreconnect>`_ for more details about this.

.. note::
    Pyro has to do a lookup every time it needs to connect one of these PYRONAME uris.
    If you connect/disconnect many times or with many different objects,
    consider using PYRO uris (you can type them directly or create them by resolving as explained in the
    following paragraph) or call :meth:`Pyro5.core.Proxy._pyroBind()` on the proxy to
    bind it to a fixed PYRO uri instead.


.. index:: PYROMETA protocol type
.. _nameserver-pyrometa:

The PYROMETA protocol type
==========================
Next to the ``PYRONAME`` protocol type there is another 'magic' protocol ``PYROMETA``.
This protocol type tells Pyro to treat the URI as metadata tags, and Pyro will
ask the name server for any (randomly chosen) object that has the given metadata tags.
The form of a PYROMETA uri is::

    PYROMETA:metatag
    PYROMETA:metatag1,metatag2,metatag3
    PYROMETA:metatag@nshostname           # with optional host name
    PYROMETA:metatag@nshostname:nsport    # with optional host name + port

So you can write this to connect to any random printer (given that all Pyro objects representing a printer
have been registered in the name server with the ``resource.printer`` metadata tag)::

    proxy=Pyro5.client.Proxy("PYROMETA:resource.printer")
    proxy.printstuff()

You have to explicitly add metadata tags when registering objects with the name server, see :ref:`nameserver-yellowpages`.
Objects without metadata tags cannot be found via ``PYROMETA`` obviously.
Note that the name server supports more advanced metadata features than what ``PYROMETA`` provides:
in a PYROMETA uri you cannot use white spaces, and you cannot ask for an object that has one or more
of the given tags -- multiple tags means that the object must have all of them.

Metadata tags can be listed if you query the name server for registrations.



.. index:: resolving object names, PYRONAME protocol type

Resolving object names
======================
'Resolving an object name' means to look it up in the name server's registry and getting
the actual URI that belongs to it (with the actual object name or id and the location of
the daemon in which it is running). This is not normally needed in user code (Pyro takes
care of it automatically for you), but it can still be useful in certain situations.

So, resolving a logical name can be done in several ways:

#. The easiest way: let Pyro do it for you! Simply pass a ``PYRONAME`` URI to the proxy constructor,
   and forget all about the resolving happening under the hood::

    obj = Pyro5.client.Proxy("PYRONAME:objectname")
    obj.method()

#. obtain a name server proxy and use its ``lookup`` method (:meth:`Pyro5.nameserver.NameServer.lookup`).
   You could then use this resolved uri to get an actual proxy, or do other things with it::

    ns = Pyro5.core.locate_ns()
    uri = ns.lookup("objectname")
    # uri now is the resolved 'objectname'
    obj = Pyro5.client.Proxy(uri)
    obj.method()

#. use a ``PYRONAME`` URI and resolve it using the ``resolve`` utility function :func:`Pyro5.core.resolve` (also available as :func:`Pyro5.api.resolve`)::

    uri = Pyro5.core.resolve("PYRONAME:objectname")
    # uri now is the resolved 'objectname'
    obj = Pyro5.client.Proxy(uri)
    obj.method()

#. use a ``PYROMETA`` URI and resolve it using the ``resolve`` utility function :func:`Pyro5.core.resolve` (also available as :func:`Pyro5.api.resolve`)::

    uri = Pyro5.core.resolve("PYROMETA:metatag1,metatag2")
    # uri is now randomly chosen from all objects having the given meta tags
    obj = Pyro5.client.Proxy(uri)


.. index::
    double: name server; registering objects
    double: name server; unregistering objects

.. _nameserver-registering:

Registering object names
========================
'Registering an object' means that you associate the URI with a logical name, so that
clients can refer to your Pyro object by using that name.
Your server has to register its Pyro objects with the name server. It first registers an
object with the Daemon, gets an URI back, and then registers that URI in the name server using
the following method on the name server proxy:

.. py:method:: register(name, uri, safe=False)

    Registers an object (uri) under a logical name in the name server.

    :param name: logical name that the object will be known as
    :type name: string
    :param uri: the URI of the object (you get it from the daemon)
    :type uri: string or :class:`Pyro5.core.URI`
    :param safe: normally registering the same name twice silently overwrites the old registration. If you set safe=True, the same name cannot be registered twice.
    :type safe: bool

You can unregister objects as well using the :py:meth:`unregister` method.
The name server also supports automatically checking for registrations that are no longer available,
for instance because the server process crashed or a network problem occurs. It will then automatically
remove those registrations after a certain timeout period.
This feature is disabled by default (it potentially requires the NS to periodically create a lot of
network connections to check for each of the registrations if it is still available). You can enable it
by setting the ``NS_AUTOCLEAN`` config item to a non zero value; it then specifies the recurring period
in seconds for the nameserver to check all its registrations. Choose an appropriately large value, the minimum
allowed is 3.


.. index:: scaling Name Server connections

Free connections to the NS quickly
==================================
By default the Name server uses a Pyro socket server based on whatever configuration is the default.
Usually that will be a threadpool based server with a limited pool size. If more clients connect to
the name server than the pool size allows, they will get a connection error.

It is suggested you apply the following pattern when using the name server in your code:

#. obtain a proxy for the NS
#. look up the stuff you need, store it
#. free the NS proxy (See :ref:`client_cleanup`)
#. use the uri's/proxies you've just looked up

This makes sure your client code doesn't consume resources in the name server for an excessive amount of time,
and more importantly, frees up the limited connection pool to let other clients get their turn.
If you have a proxy to the name server and you let it live for too long, it may eventually deny
other clients access to the name server because its connection pool is exhausted. So if you don't need
the proxy anymore, make sure to free it up.

There are a number of things you can do to improve the matter on the side of the Name Server itself.
You can control its behavior by setting certain Pyro config items before starting the server:

- You can set ``SERVERTYPE=multiplex`` to create a server that doesn't use a limited connection (thread) pool,
  but multiplexes as many connections as the system allows. However, the actual calls to the server must
  now wait on eachother to complete before the next call is processed. This may impact performance in other ways.
- You can set ``THREADPOOL_SIZE`` to an even larger number than the default.
- You can set ``COMMTIMEOUT`` to a certain value, which frees up unused connections after the given time.
  But the client code may now crash with a TimeoutError or ConnectionClosedError when it tries to use a
  proxy it obtained earlier. (You can use Pyro's autoreconnect feature to work around this but it makes
  the code more complex)


.. index::
    double: name server; Yellow-pages
    double: name server; Metadata

.. _nameserver-yellowpages:

Yellow-pages ability of the Name Server (metadata tags)
=======================================================
You can tag object registrations in the name server with one or more Metadata tags.
These are simple strings but you're free to put anything you want in it. One way of using it, is to provide
a form of Yellow-pages object lookup: instead of directly asking for the registered object by its unique name
(telephone book), you're asking for any registration from a certain *category*. You get back a list of
registered objects from the queried category, from which you can then choose the one you want.

.. note::
    Metadata tags are case-sensitive.

As an example, imagine the following objects registered in the name server (with the metadata as shown):

=================== ======================= ========
Name                Uri                     Metadata
=================== ======================= ========
printer.secondfloor PYRO:printer1@host:1234 printer
printer.hallway     PYRO:printer2@host:1234 printer
storage.diskcluster PYRO:disks1@host:1234   storage
storage.ssdcluster  PYRO:disks2@host:1234   storage
=================== ======================= ========

Instead of having to know the exact name of a required object you can query the name server for
all objects having a certain set of metadata.
So in the above case, your client code doesn't have to 'know' that it needs to lookup the ``printer.hallway``
object to get the uri of a printer (in this case the one down in the hallway).
Instead it can just ask for a list of all objects having the ``printer`` metadata tag.
It will get a list containing both ``printer.secondfloor`` and ``printer.hallway`` so you will still
have to choose the object you want to use - or perhaps even use both.
The objects tagged with ``storage`` won't be returned.

Arguably the most useful way to deal with the metadata is to use it for Yellow-pages style lookups.
You can ask for all objects having some set of metadata tags, where you can choose if
they should have *all* of the given tags or only *any one* (or more) of the given tags. Additional or
other filtering must be done in the client code itself.
So in the above example, querying with ``meta_any={'printer', 'storage'}`` will return all four
objects, while querying with ``meta_all={'printer', 'storage'}`` will return an empty list (because
there are no objects that are both a printer and storage).

**Setting metadata in the name server**

Object registrations in the name server by default have an empty set of metadata tags associated with them.
However the ``register`` method (:meth:`Pyro5.nameserver.NameServer.register`) has an optional ``metadata`` argument,
you can set that to a set of strings that will be the metadata tags associated with the object registration.
For instance::

    ns.register("printer.secondfloor", "PYRO:printer1@host:1234", metadata={"printer"})


**Getting metadata back from the name server**

The ``lookup`` (:meth:`Pyro5.nameserver.NameServer.lookup`) and ``list`` (:meth:`Pyro5.nameserver.NameServer.list`) methods
of the name server have an optional ``return_metadata`` argument.
By default it is False, and you just get back the registered URI (lookup) or a dictionary with the registered
names and their URI as values (list). If you set it to True however, you'll get back tuples instead:
(uri, set-of-metadata-tags)::

    ns.lookup("printer.secondfloor", return_metadata=True)
    # returns: (<Pyro5.core.URI at 0x6211e0, PYRO:printer1@host:1234>, {'printer'})

    ns.list(return_metadata=True)
    # returns something like:
    #   {'printer.secondfloor': ('PYRO:printer1@host:1234', {'printer'}),
    #    'Pyro.NameServer': ('PYRO:Pyro.NameServer@localhost:9090', {'class:Pyro5.nameserver.NameServer'})}
    # (as you can see the name server itself has also been registered with a metadata tag)

**Querying on metadata (Yellow-page lookup)**

You can ask the name server to list all objects having some set of metadata tags.
The ``yplookup`` (:meth:`Pyro5.nameserver.NameServer.yplookup`) method of the name server has two arguments
to allow you do do this: ``meta_all`` and ``meta_any``.

#. ``meta_all``: give all objects having *all* of the given metadata tags::

    ns.yplookup(meta_all={"printer"})
    # returns: {'printer.secondfloor': 'PYRO:printer1@host:1234'}
    ns.yplookup(meta_all={"printer", "communication"})
    # returns: {}   (there is no object that's both a printer and a communication device)

#. ``meta_any``: give all objects having *one* (or more) of the given metadata tags::

    ns.yplookup(meta_any={"storage", "printer", "communication"})
    # returns: {'printer.secondfloor': 'PYRO:printer1@host:1234'}


**Querying on metadata via ``PYROMETA`` uri (Yellow-page lookup in uri)**

As a convenience, similar to the ``PYRONAME`` uri protocol, you can use the ``PYROMETA`` uri protocol
to let Pyro do the lookup for you. It only supports ``meta_all`` lookup, but it allows you to
conveniently get a proxy like this::

    Pyro5.client.Proxy("PYROMETA:resource.printer,performance.fast")

this will connect to a (randomly chosen) object with both the ``resource.printer`` and ``performance.fast`` metadata tags.
Also see :ref:`nameserver-pyrometa`.


You can find some code that uses the metadata API in the `ns-metadata example <https://github.com/irmen/Pyro5/tree/master/examples/ns-metadata>`_ .
Note that the ``nsc`` tool (:ref:`nameserver-nsc`) also allows you to manipulate the metadata in the name server from the command line.


.. index:: Name Server API

Other methods in the Name Server API
====================================
The name server has a few other methods that might be useful at times.
For instance, you can ask it for a list of all registered objects.
Because the name server itself is a regular Pyro object, you can access its methods
through a regular Pyro proxy, and refer to the description of the exposed class to
see what methods are available: :class:`Pyro5.nameserver.NameServer`.
