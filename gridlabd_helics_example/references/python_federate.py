# python_federate.py
import helics as h
import time

fedinfo = h.helicsCreateFederateInfo()
h.helicsFederateInfoSetCoreName(fedinfo, "pyfed")
h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
h.helicsFederateInfoSetCoreInitString(fedinfo, "--federates=1")
h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_delta, 60)
fed = h.helicsCreateValueFederate("py_federate", fedinfo)

sub = h.helicsFederateRegisterSubscription(fed, "voltage_pub", "double")
h.helicsFederateEnterExecutingMode(fed)

granted_time = 0
end_time = 300

while granted_time < end_time:
    granted_time = h.helicsFederateRequestTime(fed, end_time)
    if h.helicsInputIsUpdated(sub):
        val = h.helicsInputGetDouble(sub)
        print(f"[t={granted_time}] Voltage received: {val:.2f}")

h.helicsFederateFinalize(fed)
h.helicsCloseLibrary()

