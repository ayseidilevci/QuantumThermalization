from backend.qi_manager import QuantumInspireManager

from qiskit import QuantumCircuit


def test_login(manager):

    def login_finished(return_code):
        if return_code == 0:
            print("Login successful!")
            # Continue UI flow
        else:
            print("Login failed!")

    manager.login(login_finished)

    if manager.check_login():
        print("login in checked correctly!")

def test_provider(manager):
    manager.set_provider()

def submit_testing_circut(user, backend):
    # Create a basic Bell State circuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    # Show a representation of the quantum circuit:
    print(qc)

    # Run the circuit on Quantum Inspire's platform:
    job = user.submit(qc, backend)

    # Print the results.
    result = job.result()
    print(result.get_counts())

if __name__ == "__main__":

    manager = QuantumInspireManager()

    # print("login test!")
    # test_login(manager)

    print("provider test!")
    test_provider(manager)
