.. index:: server code

*****************************
Servers: hosting Pyro objects
*****************************

This chapter explains how you write code that publishes objects to be remotely accessible.
These objects are then called *Pyro objects* and the program that provides them,
is often called a *server* program.

(The program that calls the objects is usually called the *client*.
Both roles can be mixed in a single program.)

Make sure you are familiar with Pyro's :ref:`keyconcepts` before reading on.

.. seealso::

    :doc:`config` for several config items that you can use to tweak various server side aspects.


.. index::
    single: decorators
    single: @Pyro5.server.expose
    single: @Pyro5.server.oneway
    double: decorator; expose
    double: decorator; oneway


.. _decorating-pyro-class:

Creating a Pyro class and exposing its methods and properties
=============================================================

Exposing classes, methods and properties is done using the ``@Pyro5.server.expose`` decorator.
It lets you mark the following items to be available for remote access:

- methods (including classmethod and staticmethod). You cannot expose a 'private' method, i.e. name starting with underscore.
  You *can* expose a 'dunder' method with double underscore for example ``__len__``. There is a short list of dunder methods that
  will never be remoted though (because they are essential to let the Pyro proxy function correctly).
  Make sure you put the ``@expose`` decorator after other decorators on the method, if any.
- properties (these will be available as remote attributes on the proxy) It's not possible to expose a 'private' property
  (name starting with underscore). You can't expose attributes directly. It is required to provide a @property for them
  and decorate that with ``@expose``, if you want to provide a remotely accessible attribute.
- classes as a whole (exposing a class has the effect of exposing every nonprivate method and property of the class automatically)

Anything that isn't decorated with ``@expose`` is not remotely accessible.

.. important:: **Private methods and attributes**:
    In the spirit of being secure by default, Pyro doesn't allow remote access to anything of your class unless
    explicitly told to do so. It will never allow remote access to 'private' methods and attributes
    (where 'private' means that their name starts with a single or double underscore).
    There's a special exception for the regular 'dunder' names with double underscores such as ``__len__`` though.

Here's a piece of example code that shows how a partially exposed Pyro class may look like::

    import Pyro5.server

    class PyroService(object):

        value = 42                  # not exposed

        def __dunder__(self):       # exposed
            pass

        def _private(self):         # not exposed
            pass

        def __private(self):        # not exposed
            pass

        @Pyro5.server.expose
        def get_value(self):        # exposed
            return self.value

        @Pyro5.server.expose
        @property
        def attr(self):             # exposed as 'proxy.attr' remote attribute
            return self.value

        @Pyro5.server.expose
        @attr.setter
        def attr(self, value):      # exposed as 'proxy.attr' writable
            self.value = value


.. index:: oneway decorator

**Specifying one-way methods using the @Pyro5.server.oneway decorator:**

You decide on the class of your Pyro object on the server, what methods are to be called as one-way.
You use the ``@Pyro5.server.oneway`` decorator on these methods to mark them for Pyro.
When the client proxy connects to the server it gets told automatically what methods are one-way,
you don't have to do anything on the client yourself. Any calls your client code makes on the proxy object
to methods that are marked with ``@Pyro5.server.oneway`` on the server, will happen as one-way calls::

    import Pyro5

    @Pyro5.server.expose
    class PyroService(object):

        def normal_method(self, args):
            result = do_long_calculation(args)
            return result

        @Pyro5.server.oneway
        def oneway_method(self, args):
            result = do_long_calculation(args)
            # no return value, cannot return anything to the client


See :ref:`oneway-calls-client` for the documentation about how client code handles this.
See the `oneway example <https://github.com/irmen/Pyro5/tree/master/examples/oneway>`_ for some code that demonstrates the use of oneway methods.


Exposing classes and methods without changing existing source code
==================================================================

In the case where you cannot or don't want to change existing source code,
it's not possible to use the ``@expose`` decorator to tell Pyro what methods should be exposed.
This can happen if you're dealing with third-party library classes or perhaps a generic module that
you don't want to 'taint' with a Pyro dependency because it's used elsewhere too.

There are a few possibilities to deal with this:

**Use adapter classes**

The preferred solution is to not use the classes from the third party library directly, but create an adapter class yourself
with the appropriate ``@expose`` set on it or on its methods. Register this adapter class instead.
Then use the class from the library from within your own adapter class.
This way you have full control over what exactly is exposed, and what parameter and return value types
travel over the wire.

**Create exposed classes by using ``@expose`` as a function**

Creating adapter classes is good but if you're looking for the most convenient solution we can do better.
You can still use ``@expose`` to make a class a proper Pyro class with exposed methods,
*without having to change the source code* due to adding @expose decorators, and without having
to create extra classes yourself.
Remember that Python decorators are just functions that return another function (or class)? This means you can also
call them as a regular function yourself, which allows you to use classes from third party libraries like this::

    from awesome_thirdparty_library import SomeClassFromLibrary
    import Pyro5.server

    # expose the class from the library using @expose as wrapper function:
    ExposedClass = Pyro5.server.expose(SomeClassFromLibrary)

    daemon.register(ExposedClass)    # register the exposed class rather than the library class itself


There are a few caveats when using this:

#. You can only expose the class and all its methods as a whole, you can't cherrypick methods that should be exposed

#. You have no control over what data is returned from the methods. It may still be required to deal with
   serialization issues for instance when a method of the class returns an object whose type is again a class from the library.


See the `thirdpartylib example <https://github.com/irmen/Pyro5/tree/master/examples/thirdpartylib>`_ for a little server that deals with such a third party library.


.. index:: publishing objects

.. _publish-objects:

Pyro Daemon: publishing Pyro objects
====================================

To publish a regular Python object and turn it into a Pyro object,
you have to tell Pyro about it. After that, your code has to tell Pyro to start listening for incoming
requests and to process them. Both are handled by the *Pyro daemon*.

In its most basic form, you create one or more classes that you want to publish as Pyro objects,
you create a daemon, register the class(es) with the daemon, and then enter the daemon's request loop::

    import Pyro5.server

    @Pyro5.server.expose
    class MyPyroThing(object):
        # ... methods that can be called go here...
        pass

    daemon = Pyro5.server.Daemon()
    uri = daemon.register(MyPyroThing)
    print(uri)
    daemon.requestLoop()

Once a client connects, Pyro will create an instance of the class and use that single object
to handle the remote method calls during one client proxy session. The object is removed once
the client disconnects. Another client will cause another instance to be created for its session.
You can control more precisely when, how, and for how long Pyro will create an instance of your Pyro class.
See :ref:`server-instancemode` below for more details.

Anyway, when you run the code printed above, the uri will be printed and the server sits waiting for requests.
The uri that is being printed looks a bit like this: ``PYRO:obj_dcf713ac20ce4fb2a6e72acaeba57dfd@localhost:51850``
Client programs use these uris to access the specific Pyro objects.

.. note::
    From the address in the uri that was printed you can see that Pyro by default binds its daemons on localhost.
    This means you cannot reach them from another machine on the network (a security measure).
    If you want to be able to talk to the daemon from other machines, you have to
    explicitly provide a hostname to bind on. This is done by giving a ``host`` argument to
    the daemon, see the paragraphs below for more details on this.

.. index:: private methods

.. note:: **Private methods:**
    Pyro considers any method or attribute whose name starts with at least one underscore ('_'), private.
    These cannot be accessed remotely.
    An exception is made for the 'dunder' methods with double underscores, such as ``__len__``. Pyro follows
    Python itself here and allows you to access these as normal methods, rather than treating them as private.

.. note::
    You can publish any regular Python object as a Pyro object.
    However since Pyro adds a few Pyro-specific attributes to the object, you can't use:

    * types that don't allow custom attributes, such as the builtin types (``str`` and ``int`` for instance)
    * types with ``__slots__`` (a possible way around this is to add Pyro's custom attributes to your ``__slots__``, but that isn't very nice)

.. note::
    Most of the the time a Daemon will keep running. However it's still possible to nicely free its resources
    when the request loop terminates by simply using it as a context manager in a ``with`` statement, like so::

        with Pyro5.server.Daemon() as daemon:
            daemon.register(...)
            daemon.requestLoop()


.. index:: publishing objects oneliner, serve
.. _server-servesimple:

Oneliner Pyro object publishing: Pyro5.server.serve()
-----------------------------------------------------
Ok not really a one-liner, but one statement: use :py:meth:`serve` to publish a dict of objects/classes and start Pyro's request loop.
The code above could also be written as::

    import Pyro5.server

    @Pyro5.server.expose
    class MyPyroThing(object):
        pass

    obj = MyPyroThing()
    Pyro5.server.serve(
        {
            MyPyroThing: None,    # register the class
            obj: None             # register one specific instance
        },
        ns=False)

You can perform some limited customization:

.. py:method:: serve(objects [host=None, port=0, daemon=None, use_ns=True, verbose=True])

    Very basic method to fire up a daemon that hosts a bunch of objects.
    The objects will be registered automatically in the name server if you specify this.
    API reference: :py:func:`Pyro5.server.serve`

    :param objects: mapping of objects/classes to names, these are the Pyro objects that will be hosted by the daemon, using the names you provide as values in the mapping.
        Normally you'll provide a name yourself but in certain situations it may be useful to set it to ``None``. Read below for the exact behavior there.
    :type objects: dict
    :param host: optional hostname where the daemon should be reached on. Details below at :ref:`create_deamon`
    :type host: str or None
    :param port: optional port number where the daemon should be accessible on
    :type port: int
    :param daemon: optional existing daemon to use, that you created yourself.
        If you don't specify this, the method will create a new daemon object by itself.
    :type daemon: Pyro5.server.Daemon
    :param use_ns: optional, if True (the default), the objects will also be registered in the name server (located using :py:meth:`Pyro5.core.locate_ns`) for you.
        If this parameters is False, your objects will only be hosted in the daemon and are not published in a name server.
        Read below about the exact behavior of the object names you provide in the ``objects`` dictionary.
    :type ns: bool
    :param verbose: optional, if True (the default), print out a bit of info on the objects that are registered
    :type verbose: bool
    :returns: nothing, it starts the daemon request loop and doesn't return until that stops.

If you set ``use_ns=True`` (the default) your objects will appear in the name server as well.
Usually this means you provide a logical name for every object in the ``objects`` dictionary.
If you don't (= set it to ``None``), the object will still be available in the daemon (by a generated name) but will *not* be registered
in the name server (this is a bit strange, but hey, maybe you don't want all the objects to be visible in the name server).

When not using a name server at all (``use_ns=False``), the names you provide are used as the object names
in the daemon itself. If you set the name to ``None`` in this case, your object will get an automatically generated internal name,
otherwise your own name will be used.

.. important::
    - The names you provide for each object have to be unique (or ``None``). For obvious reasons you can't register multiple objects with the same names.
    - if you use ``None`` for the name, you have to use the ``verbose`` setting as well, otherwise you won't know the name that Pyro generated for you.
      That would make your object more or less unreachable.

The uri that is used to register your objects in the name server with, is of course generated by the daemon.
So if you need to influence that, for instance because of NAT/firewall issues,
it is the daemon's configuration you should be looking at.

If you don't provide a daemon yourself, :py:meth:`serve` will create a new one for you using the default configuration or
with a few custom parameters you can provide in the call, as described above.
If you don't specify the ``host`` and ``port`` parameters, it will simple create a Daemon using the default settings.
If you *do* specify ``host`` and/or ``port``, it will use these as parameters for creating the Daemon (see next paragraph).
If you need to further tweak the behavior of the daemon, you have to create one yourself first, with the desired
configuration. Then provide it to this function using the ``daemon`` parameter. Your daemon will then be used instead of a new one::

    custom_daemon = Pyro5.server.Daemon(host="example", nathost="example")    # some additional custom configuration
    Pyro5.server.serve(
        {
            MyPyroThing: None
        },
        daemon = custom_daemon)


.. index::
    double: Pyro daemon; creating a daemon

.. _create_deamon:

Creating a Daemon
-----------------
Pyro's daemon is ``Pyro5.server.Daemon``.
It has a few optional arguments when you create it:


.. function:: Daemon([host=None, port=0, unixsocket=None, nathost=None, natport=None, interface=DaemonObject, connected_socket=None])

    Create a new Pyro daemon.

    :param host: the hostname or IP address to bind the server on. Default is ``None`` which means it uses the configured default (which is localhost).
                 It is necessary to set this argument to a visible hostname or ip address, if you want to access the daemon from other machines.
                 When binding to a hostname be careful of your OS's policies as it might still bind to localhost as well. Depending on your DNS
                 setup you may have to use "", "0.0.0.0" or an explicit externally visible IP addres to make the server accessible over the network.
    :type host: str or None
    :param port: port to bind the server on. Defaults to 0, which means to pick a random port.
    :type port: int
    :param unixsocket: the name of a Unix domain socket to use instead of a TCP/IP socket. Default is ``None`` (don't use).
    :type unixsocket: str or None
    :param nathost: hostname to use in published addresses (useful when running behind a NAT firewall/router). Default is ``None`` which means to just use the normal host.
                    For more details about NAT, see :ref:`nat-router`.
    :type host: str or None
    :param natport: port to use in published addresses (useful when running behind a NAT firewall/router). If you use 0 here,
                    Pyro will replace the NAT-port by the internal port number to facilitate one-to-one NAT port mappings.
    :type port: int
    :param interface: optional alternative daemon object implementation (that provides the Pyro API of the daemon itself)
    :type interface: Pyro5.server.DaemonObject
    :param connected_socket: optional existing socket connection to use instead of creating a new server socket
    :type interface: socket


.. index::
    double: Pyro daemon; registering objects/classes

Registering objects/classes
---------------------------
Every object you want to publish as a Pyro object needs to be registered with the daemon.
You can let Pyro choose a unique object id for you, or provide a more readable one yourself.

.. method:: Daemon.register(obj_or_class [, objectId=None, force=False, weak=False])

    Registers an object with the daemon to turn it into a Pyro object.

    :param obj_or_class: the singleton instance or class to register (class is the preferred way)
    :param objectId: optional custom object id (must be unique). Default is to let Pyro create one for you.
    :type objectId: str or None
    :param force: optional flag to force registration, normally Pyro checks if an object had already been registered.
        If you set this to True, the previous registration (if present) will be silently overwritten.
    :param weak: only store weak reference to the object, automatically unregistering it when it is garbage-collected. Without this, the daemon will keep the object alive by having it stored in its mapping, preventing garbage-collection until manual unregistration.
    :type force: bool
    :returns: an uri for the object
    :rtype: :class:`Pyro5.core.URI`

It is important to do something with the uri that is returned: it is the key to access the Pyro object.
You can save it somewhere, or perhaps print it to the screen.
The point is, your client programs need it to be able to access your object (they need to create a proxy with it).

Maybe the easiest thing is to store it in the Pyro name server.
That way it is almost trivial for clients to obtain the proper uri and connect to your object.
See :doc:`nameserver` for more information (:ref:`nameserver-registering`), but it boils down to
getting a name server proxy and using its ``register`` method::

    uri = daemon.register(some_object)
    ns = Pyro5.core.locate_ns()
    ns.register("example.objectname", uri)


.. note::
    If you ever need to create a new uri for an object, you can use :py:meth:`Pyro5.server.Daemon.uriFor`.
    The reason this method exists on the daemon is because an uri contains location information and
    the daemon is the one that knows about this.

Intermission: Example 1: server and client not using name server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A little code example that shows the very basics of creating a daemon and publishing a Pyro object with it.
Server code::

    import Pyro5.server

    @Pyro5.server.expose
    class Thing(object):
        def method(self, arg):
            return arg*2

    # ------ normal code ------
    daemon = Pyro5.server.Daemon()
    uri = daemon.register(Thing)
    print("uri=",uri)
    daemon.requestLoop()

    # ------ alternatively, using serve -----
    Pyro5.server.serve(
        {
            Thing: None
        },
        ns=False, verbose=True)

Client code example to connect to this object::

    import Pyro5.client
    # use the URI that the server printed:
    uri = "PYRO:obj_b2459c80671b4d76ac78839ea2b0fb1f@localhost:49383"
    thing = Pyro5.client.Proxy(uri)
    print(thing.method(42))   # prints 84

With correct additional parameters --described elsewhere in this chapter-- you can control on which port the daemon is listening,
on what network interface (ip address/hostname), what the object id is, etc.

Intermission: Example 2: server and client, with name server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A little code example that shows the very basics of creating a daemon and publishing a Pyro object with it,
this time using the name server for easier object lookup.
Server code::

    import Pyro5.server
    import Pyro5.core

    @Pyro5.server.expose
    class Thing(object):
        def method(self, arg):
            return arg*2

    # ------ normal code ------
    daemon = Pyro5.server.Daemon(host="yourhostname")
    ns = Pyro5.core.locate_ns()
    uri = daemon.register(Thing)
    ns.register("mythingy", uri)
    daemon.requestLoop()

    # ------ alternatively, using serve -----
    Pyro5.server.serve(
        {
            Thing: "mythingy"
        },
        ns=True, verbose=True, host="yourhostname")

Client code example to connect to this object::

    import Pyro5.client
    thing = Pyro5.client.Proxy("PYRONAME:mythingy")
    print(thing.method(42))   # prints 84


.. index::
    double: Pyro daemon; unregistering objects

Unregistering objects
---------------------
When you no longer want to publish an object, you need to unregister it from the daemon (unless it was registered with ``weak=True`` when it will be unregistered automatically when garbage-collected):

.. method:: Daemon.unregister(objectOrId)

    :param objectOrId: the object to unregister
    :type objectOrId: object itself or its id string


.. index:: request loop

Running the request loop
------------------------
Once you've registered your Pyro object you'll need to run the daemon's request loop to make
Pyro wait for incoming requests.

.. method:: Daemon.requestLoop([loopCondition])

    :param loopCondition: optional callable returning a boolean, if it returns False the request loop will be aborted and the call returns

This is Pyro's event loop and it will take over your program until it returns (it might never.)
If this is not what you want, you can control it a tiny bit with the ``loopCondition``, or read the next paragraph.

.. index::
    double: event loop; integrate Pyro's requestLoop

Integrating Pyro in your own event loop
---------------------------------------
If you want to use a Pyro daemon in your own program that already has an event loop (aka main loop),
you can't simply call ``requestLoop`` because that will block your program.
A daemon provides a few tools to let you integrate it into your own event loop:

* :py:attr:`Pyro5.server.Daemon.sockets` - list of all socket objects used by the daemon, to inject in your own event loop
* :py:meth:`Pyro5.server.Daemon.events` - method to call from your own event loop when Pyro needs to process requests. Argument is a list of sockets that triggered.

For more details and example code, see the
`eventloop <https://github.com/irmen/Pyro5/tree/master/examples/eventloop>`_ and
`gui_eventloop <https://github.com/irmen/Pyro5/tree/master/examples/gui_eventloop>`_ examples.
They show how to use Pyro including a name server, in your own event loop, and also possible ways
to use Pyro from within a GUI program with its own event loop.

.. index:: Combining Daemons

Combining Daemon request loops
------------------------------
In certain situations you will be dealing with more than one daemon at the same time.
For instance, when you want to run your own Daemon together with an 'embedded' Name Server Daemon,
or perhaps just another daemon with different settings.

Usually you run the daemon's :meth:`Pyro5.server.Daemon.requestLoop` method to handle incoming requests.
But when you have more than one daemon to deal with, you have to run the loops of all of them in parallel somehow.
There are a few ways to do this:

1. multithreading: run each daemon inside its own thread
2. multiplexing event loop: write a multiplexing event loop and call back into the appropriate
   daemon when one of its connections send a request.
   You can do this using :mod:`selectors` or :mod:`select` and you can even integrate other (non-Pyro)
   file-like selectables into such a loop. Also see the paragraph above.
3. use :meth:`Pyro5.server.Daemon.combine` to combine several daemons into one,
   so that you only have to call the requestLoop of that "master daemon".
   Basically Pyro will run an integrated multiplexed event loop for you.
   You can combine normal Daemon objects, the NameServerDaemon and also the name server's BroadcastServer.
   Again, have a look at the `eventloop example <https://github.com/irmen/Pyro5/tree/master/examples/eventloop>`_ to see how this can be done.
   (Note: this will only work with the ``multiplex`` server type, not with the ``thread`` type)


.. index::
    double: Pyro daemon; shutdown
    double: Pyro daemon; cleaning up

Cleaning up
-----------
To clean up the daemon itself (release its resources) either use the daemon object
as a context manager in a ``with`` statement, or manually call :py:meth:`Pyro5.server.Daemon.close`.

Of course, once the daemon is running, you first need a clean way to stop the request loop before
you can even begin to clean things up.

You can use force and hit ctrl-C or ctrl-\ or ctrl-Break to abort the request loop, but
this usually doesn't allow your program to clean up neatly as well.
It is therefore also possible to leave the loop cleanly from within your code (without using :py:meth:`sys.exit` or similar).
You'll have to provide a ``loopCondition`` that you set to ``False`` in your code when you want
the daemon to stop the loop. You could use some form of semi-global variable for this.
(But if you're using the threaded server type, you have to also set ``COMMTIMEOUT`` because otherwise
the daemon simply keeps blocking inside one of the worker threads).

Another possibility is calling  :py:meth:`Pyro5.server.Daemon.shutdown` on the running daemon object.
This will also break out of the request loop and allows your code to neatly clean up after itself,
and will also work on the threaded server type without any other requirements.

If you are using your own event loop mechanism you have to use something else, depending on your own loop.


.. index::
    single: @Pyro5.server.behavior
    instance modes; instance_mode
    instance modes; instance_creator
.. _server-instancemode:

Controlling Instance modes and Instance creation
================================================

While it is possible to register a single singleton *object* with the daemon,
it is actually preferred that you register a *class* instead.
When doing that, it is Pyro itself that creates an instance (object) when it needs it.
This allows for more control over when and for how long Pyro creates objects.

Controlling the instance mode and creation is done by decorating your class with ``Pyro5.server.behavior``
and setting its ``instance_mode`` or/and ``instance_creator`` parameters. It can only be used
on a class definition, because these behavioral settings only make sense at that level.

By default, Pyro will create an instance of your class per *session* (=proxy connection)
Here is an example of registering a class that will have one new instance for *every single method call* instead::

    import Pyro5.server

    @Pyro5.server.behavior(instance_mode="percall")
    class MyPyroThing(object):
        @Pyro5.server.expose
        def method(self):
            return "something"

    daemon = Pyro5.server.Daemon()
    uri = daemon.register(MyPyroThing)
    print(uri)
    daemon.requestLoop()

There are three possible choices for the ``instance_mode`` parameter:

- ``session``: (the default) a new instance is created for every new proxy connection, and is reused for
  all the calls during that particular proxy session. Other proxy sessions will deal with a different instance.
- ``single``: a single instance will be created and used for all method calls (for this daemon), regardless what proxy
  connection we're dealing with. This is the same as creating and registering a single object yourself
  (the old style of registering code with the deaemon). Be aware that the methods on this object can be called
  from separate threads concurrently.
- ``percall``: a new instance is created for every single method call, and discarded afterwards.


**Instance creation**

.. sidebar:: Instance creation is lazy

    When you register a class in this way, be aware that Pyro only creates an actual
    instance of it when it is first needed. If nobody connects to the deamon requesting
    the services of this class, no instance is ever created.

Normally Pyro will simply use a default parameterless constructor call to create the instance.
If you need special initialization or the class's init method requires parameters, you have to specify
an ``instance_creator`` callable as well. Pyro will then use that to create an instance of your class.
It will call it with the class to create an instance of as the single parameter.

See the `instancemode example <https://github.com/irmen/Pyro5/tree/master/examples/instancemode>`_ to learn about various ways to use this.
See the `usersession example <https://github.com/irmen/Pyro5/tree/master/examples/usersession>`_ to learn how you could use it to build user-bound resource access without concurrency problems.


.. index:: automatic proxying

Autoproxying
============
Pyro will automatically take care of any Pyro objects that you pass around through remote method calls.
It will replace them by a proxy automatically, so the receiving side can call methods on it and be
sure to talk to the remote object instead of a local copy. There is no need to create a proxy object manually.
All you have to do is to register the new object with the appropriate daemon::

    def some_pyro_method(self):
        thing=SomethingNew()
        self._pyroDaemon.register(thing)
        return thing    # just return it, no need to return a proxy

There is a `autoproxy example <https://github.com/irmen/Pyro5/tree/master/examples/autoproxy>`_ that shows the use of this feature,
and several other examples also make use of it.

Note that when using the marshal serializer, this feature doesn't work. You have to use
one of the other serializers to use autoproxying.


.. index:: concurrency model, server types, SERVERTYPE

.. _object_concurrency:

Server types and Concurrency model
==================================
Pyro supports multiple server types (the way the Daemon listens for requests). Select the
desired type by setting the ``SERVERTYPE`` config item. It depends very much on what you
are doing in your Pyro objects what server type is most suitable. For instance, if your Pyro
object does a lot of I/O, it may benefit from the parallelism provided by the thread pool server.
However if it is doing a lot of CPU intensive calculations, the multiplexed server may be more
appropriate. If in doubt, go with the default setting.

.. index::
    double: server type; threaded

1. threaded server (servertype ``"thread"``, this is the default)
    This server uses a dynamically adjusted thread pool to handle incoming proxy connections.
    If the max size of the thread pool is too small for the number of proxy connections, new proxy connections
    will fail with an exception.
    The size of the pool is configurable via some config items:

        - ``THREADPOOL_SIZE``         this is the maximum number of threads that Pyro will use
        - ``THREADPOOL_SIZE_MIN``     this is the minimum number of threads that must remain standby

    Every proxy on a client that connects to the daemon will be assigned to a thread to handle
    the remote method calls. This way multiple calls can potentially be processed concurrently.
    *This means your Pyro object may have to be made thread-safe*!
    If you registered the pyro object's class with instance mode ``single``, that single instance
    will be called concurrently from different threads. If you used instance mode ``session`` or ``percall``,
    the instance will not be called from different threads because a new one is made per connection or even per call.
    But in every case, if you access a shared resource from your Pyro object,
    you may need to take thread locking measures such as using Queues.


.. index::
    double: server type; multiplex

2. multiplexed server (servertype ``"multiplex"``)
    This server uses a connection multiplexer to process
    all remote method calls sequentially. No threads are used in this server.
    It uses the best supported selector available on your platform (kqueue, poll, select).
    It means only one method call is running at a time, so if it takes a while to complete, all other
    calls are waiting for their turn (even when they are from different proxies).
    The instance mode used for registering your class, won't change the way
    the concurrent access to the instance is done: in all cases, there is only one call active at all times.
    Your objects will never be called concurrently from different threads, because there are no threads.
    It does still affect when and how often Pyro creates an instance of your class.

.. note::
    If the ``ONEWAY_THREADED`` config item is enabled (it is by default), *oneway* method calls will
    be executed in a separate worker thread, regardless of the server type you're using.

.. index::
    double: server type; what to choose?

*When to choose which server type?*
With the threadpool server at least you have a chance to achieve concurrency, and
you don't have to worry much about blocking I/O in your remote calls. The usual
trouble with using threads in Python still applies though:
Python threads don't run concurrently unless they release the :abbr:`GIL (Global Interpreter Lock)`.
If they don't, you will still hang your server process.
For instance if a particular piece of your code doesn't release the :abbr:`GIL (Global Interpreter Lock)` during
a longer computation, the other threads will remain asleep waiting to acquire the :abbr:`GIL (Global Interpreter Lock)`. One of these threads may be
the Pyro server loop and then your whole Pyro server will become unresponsive.
Doing I/O usually means the :abbr:`GIL (Global Interpreter Lock)` is released.
Some C extension modules also release it when doing their work. So, depending on your situation, not all hope is lost.

With the multiplexed server you don't have threading problems: everything runs in a single main thread.
This means your requests are processed sequentially, but it's easier to make the Pyro server
unresponsive. Any operation that uses blocking I/O or a long-running computation will block
all remote calls until it has completed.

.. index::
    double: server; serialization

Serialization
=============
Pyro will serialize the objects that you pass to the remote methods, so they can be sent across
a network connection. Depending on the serializer that is being used for your Pyro server,
there will be some limitations on what objects you can use, and what serialization format is
required of the clients that connect to your server.

If your server also uses Pyro client code/proxies, you might also need to
select the serializer for these by setting the ``SERIALIZER`` config item.

See the :doc:`/config` chapter for details about the config items.
See :ref:`object-serialization` for more details about serialization and the new config items.


Other features
==============

.. index:: attributes added to Pyro objects

Attributes added to Pyro objects
--------------------------------
The following attributes will be added to your object if you register it as a Pyro object:

* ``_pyroId`` - the unique id of this object (a ``str``)
* ``_pyroDaemon`` - a reference to the :py:class:`Pyro5.server.Daemon` object that contains this object

Even though they start with an underscore (and are private, in a way),
you can use them as you so desire. As long as you don't modify them!
The daemon reference for instance is useful to register newly created objects with,
to avoid the need of storing a global daemon object somewhere.


These attributes will be removed again once you unregister the object.

.. index:: network adapter binding, IP address, localhost, 127.0.0.1

Network adapter binding and localhost
-------------------------------------

All Pyro daemons bind on localhost by default. This is because of security reasons.
This means only processes on the same machine have access to your Pyro objects.
If you want to make them available for remote machines, you'll have to tell Pyro on what
network interface address it must bind the daemon.
This also extends to the built in servers such as the name server.

.. warning::
    Read chapter :doc:`security` before exposing Pyro objects to remote machines!

There are a few ways to tell Pyro what network address it needs to use.
You can set a global config item ``HOST``, or pass a ``host`` parameter to the constructor of a Daemon,
or use a command line argument if you're dealing with the name server.
For more details, refer to the chapters in this manual about the relevant Pyro components.

Pyro provides a couple of utility functions to help you with finding the appropriate IP address
to bind your servers on if you want to make them publicly accessible:

* :py:func:`Pyro5.socketutil.get_ip_address`
* :py:func:`Pyro5.socketutil.get_interface`


Cleaning up / disconnecting stale client connections
----------------------------------------------------
A client proxy will keep a connection open even if it is rarely used.
It's good practice for the clients to take this in consideration and release the proxy.
But the server can't enforce this, some clients may keep a connection open for a long time.
Unfortunately it's hard to tell when a client connection has become stale (unused).
Pyro's default behavior is to accept this fact and not kill the connection.
This does mean however that many stale client connections will eventually block the
server's resources, for instance all workers threads in the threadpool server.

There's a simple possible solution to this, which is to specify a communication timeout
on your server. For more information about this, read :ref:`tipstricks_release_proxy`.


.. index:: Daemon API

Daemon Pyro interface
---------------------
A rather interesting aspect of Pyro's Daemon is that it (partly) is a Pyro object itself.
This means it exposes a couple of remote methods that you can also invoke yourself if you want.
The object exposed is :class:`Pyro5.server.DaemonObject` (as you can see it is a bit limited still).

You access this object by creating a proxy for the ``"Pyro.Daemon"`` object. That is a reserved
object name. You can use it directly but it is preferable to use the constant
``Pyro5.constants.DAEMON_NAME``. An example follows that accesses the daemon object from a running name server::

    >>> import Pyro5.client
    >>> daemon=Pyro5.client.Proxy("PYRO:"+Pyro5.constants.DAEMON_NAME+"@localhost:9090")
    >>> daemon.ping()
    >>> daemon.registered()
    ['Pyro.NameServer', 'Pyro.Daemon']


Intercepting errors in user code executed in a method call
----------------------------------------------------------
When a method call is executed in a Pyro server/daemon, it eventually will execute some
user written code that implements the remote method. This user code may raise an exception
(intentionally or not). Normally, Pyro will only report the exception to the calling client.

It may be useful however to also process the error on the *server*, for instance, to log the error
somewhere for later reference. For this purpose, you can set the ``methodcall_error_handler`` attribute
on the daemon object to a custom error handler function. See the `exceptions example <https://github.com/irmen/Pyro5/tree/master/examples/exceptions>`_ .
This function's signature is::

    def custom_error_handler(daemon: Daemon, client_sock: socketutil.SocketConnection,
                             method: Callable, vargs: Sequence[Any], kwargs: Dict[str, Any],
                             exception: Exception) -> None

