import helics as h
import json
import time

def initialize_federate():
    fed = h.helicsCreateCombinationFederateFromConfig("federate1.json")
    pub = h.helicsFederateGetPublication(fed, "storage_powers")
    return fed, pub


def execute_federate(fed):
    h.helicsFederateEnterExecutingMode(fed)

def publish_values(pub, fed):
    print("Publishing simple JSON dict...")
    for t in range(0, 5):
        storage_data = {"bat_1": 10.0 + t, "bat_2": 20.0 + t}
        json_str = json.dumps(storage_data)
        h.helicsPublicationPublishString(pub, json_str)
        granted_time = h.helicsFederateRequestTime(fed, (t + 1) * 60)
        print(f"Published at t={granted_time}: {json_str}")
        time.sleep(0.1)

def disconnect_federate(fed):
    h.helicsFederateDisconnect(fed)
    print("federate1 done.")

if __name__ == "__main__":
    fed, pub = initialize_federate()
    execute_federate(fed)
    publish_values(pub, fed)
    disconnect_federate(fed)
    print("Federate1 execution completed.")