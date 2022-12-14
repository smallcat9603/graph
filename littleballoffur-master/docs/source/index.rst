:github_url: https://github.com/benedekrozemberczki/littleballoffur

Little Ball of Fur Documentation
==================================

*Little Ball of Fur* consists of methods to do sampling of graph structured data. To put it simply it is a Swiss Army knife for graph sampling tasks. First, it includes a large variety of vertex, edge and expansions sampling techniques. Second, it provides a unified application public interface which makes the application of sampling algorithms trivial for end-users. Implemented methods cover a wide range of networking (`Networking <https://link.springer.com/conference/networking>`_, `INFOCOM <https://infocom2020.ieee-infocom.org/>`_, `SIGCOMM  <http://www.sigcomm.org/>`_) and data mining (`KDD <https://www.kdd.org/kdd2020/>`_, `TKDD <https://dl.acm.org/journal/tkdd>`_, `ICDE <http://www.wikicfp.com/cfp/program?id=1331&s=ICDE&f=International%20Conference%20on%20Data%20Engineering>`_) conferences, workshops, and pieces from prominent journals.

.. code-block:: latex

    >@inproceedings{rozemberczki2020little,
         title={{Little Ball of Fur: A Python Library for Graph Sampling}},
         author={Benedek Rozemberczki and Oliver Kiss and Rik Sarkar},
         year={2020},
         pages={3133–3140},
         booktitle={Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20)},
         organization={ACM},
    }


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/introduction
   notes/create_dataset
   notes/examples
   notes/resources

.. toctree::
   :glob:
   :maxdepth: 0
   :caption: Package Reference

   modules/node_sampling
   modules/edge_sampling
   modules/exploration_sampling
   modules/dataset
