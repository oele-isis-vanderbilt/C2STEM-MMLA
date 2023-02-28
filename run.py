import pathlib
import os
import sys

# Third-party Imports
import chimerapy as cp
cp.debug([])

# Internal Imports
import c2mmla

# Constant
GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent
DATA_DIR = GIT_ROOT/'data'/'KinectData'
SESSION_VIDEOS = {
    'OELE01': DATA_DIR/'OELE01'/'2022-10-05--11-52-55',
    'OELE02': DATA_DIR/'OELE02'/'2022-10-05--11-48-33',
    'OELE08': DATA_DIR/'OELE08'/'2022-10-05--11-52-02'
}
assert all([v.exists() for v in SESSION_VIDEOS.values()])

def single_video(manager: cp.Manager):

    graph = cp.Graph()
    v1 = c2mmla.KinectNode(name='v1', kinect_data_folder=SESSION_VIDEOS['OELE01'])
    graph.add_nodes_from([v1])
    mapping = {'local': ['v1']}

    manager.commit_graph(
        graph=graph,
        mapping=mapping,
        send_packages=[{"name": "c2mmla", "path": GIT_ROOT/"c2mmla"}]
    )

def multiple_video(manager: cp.Manager):

    graph = cp.Graph()
    node_names = []
    for k, v in SESSION_VIDEOS.items():
        node = c2mmla.KinectNode(name=k, kinect_data_folder=v)
        graph.add_node(node)
        node_names.append(node.name)

    mapping = {'local': node_names}
    manager.commit_graph(
        graph=graph,
        mapping=mapping,
        send_packages=[{"name": "c2mmla", "path": GIT_ROOT/"c2mmla"}]
    )

if __name__ == "__main__":

    # Create default manager and desired graph
    manager = cp.Manager(logdir=GIT_ROOT/"runs", port=0)
    worker = cp.Worker(name="local")
    worker.connect(host=manager.host, port=manager.port)

    # Wait until workers connect
    while True:
        q = input("All workers connected? (Y/n)")
        if q.lower() == "y":
            break
        elif q.lower() == 'q':
            manager.shutdown()
            worker.shutdown()
            sys.exit(0)

    # Commit the graph
    try:
        # Configure the Manager (testing different setups)
        # single_video(manager)
        multiple_video(manager)
    except Exception as e:
        manager.shutdown()
        worker.shutdown()
        raise e

    # Wail until user stops
    while True:
        q = input("Ready to start? (Y/n)")
        if q.lower() == "y":
            break
        elif q.lower() == 'q':
            manager.shutdown()
            worker.shutdown()
            sys.exit(0)

    manager.start()

    # Wail until user stops
    while True:
        q = input("Stop? (Y/n)")
        if q.lower() == "y":
            break

    manager.stop()
    manager.collect()
    manager.shutdown()
