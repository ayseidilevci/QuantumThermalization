from io import open_code
from backend.qi_manager import QuantumInspireManager
from qi_utilities.utility_functions.data_handling import StoreProjectRecord, RetrieveProjectRecord

from qiskit import QuantumCircuit

from qiskit.visualization import plot_histogram

def test_login(manager):

    def show_code(code):
        print(code)

    def login_return(code):
        print(code)

    manager.login(on_code=show_code, on_complete=login_return)

def test_provider(manager):
    manager.set_provider()


def create_circuit():
    # Create a basic Bell State circuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def test_submit_circut(manager):
    qc = create_circuit()
    print(qc)

    job = manager.submit(qc, manager.provider.get_backend("QX emulator"))
    job_id = str(job.circuits_run_data[0].job_id)

    results = job.result()
    counts = results.data()["counts"]
    print(counts) 

    # Job Stored in ~/Documents/QuantumInspireProjects
    StoreProjectRecord(job)

    loaded_job = RetrieveProjectRecord(job_id)
    lodead_results = loaded_job.get_counts()
    print(lodead_results)

 
if __name__ == "__main__":

    manager = QuantumInspireManager()

    print("login test!")

    test_login(manager)

    # print("provider test!")
    # test_provider(manager)
    #
    # print("submit circuit test!")
    # job = test_submit_circut(manager)

