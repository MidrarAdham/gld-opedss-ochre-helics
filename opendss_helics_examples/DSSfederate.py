import helics as h
import json
import time

def initialize_federate():
    fed = h.helicsCreateCombinationFederateFromConfig("DSSfederate.json")
    sub = h.helicsFederateGetInputByIndex(fed, 0)
    return fed, sub

def execute_federate(fed):
    print("Entering executing mode...")
    h.helicsFederateEnterExecutingMode(fed)

def subscribe_values(sub, fed):
    '''
    subscribe to values from federate1.py. See federate1.py, function with the name helicsPublicationPublishString.
    
    Right now:
    - it is a simple JSON dict with two keys: bat_1 and bat_2. It is not reading from files or
    any other source, just a simple dict.

    - It does not send the values to OpenDSS, it just prints them to the console to show values were received.
    
    In the DSSfederate.log file, you'll notice that the subscribed values are printed out five times, once for each time step.
    This is because the loop in the main function of federate1.py runs five times,
    publishing a new JSON string each time.
    
    The time steps are 60 seconds apart, so the values are published at t=60, t=120, t=180, t=240, and t=300 seconds.
    '''
    for t in range(0, 5):
        granted_time = h.helicsFederateRequestTime(fed, (t + 1) * 60)
        raw = h.helicsInputGetString(sub)
        print("\n\nThis is coming from the publisher, federate1.py, and it is grabbed by helicsInputGetString")
        print("\n\n: ", raw,"\n\n")
        try:
            parsed = json.loads(raw)
            print(f"t={granted_time}: Received dict with {len(parsed)} keys, values: {parsed}")
        except Exception as e:
            print(f"t={granted_time}: JSON decode failed: {raw} → {e}")
        time.sleep(0.1)

def disconnect_federate(fed):
    h.helicsFederateDisconnect(fed)
    print("DSSfederate done.")

if __name__ == "__main__":
    fed, sub = initialize_federate()
    execute_federate(fed)
    subscribe_values(sub, fed)
    disconnect_federate(fed)
    print("DSSfederate execution completed.")
