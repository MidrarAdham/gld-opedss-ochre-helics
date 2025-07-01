So far, I resolved some HELICS issues. Here are the important ones:

- [error] broker responded with error: duplicate broker name detected

This error was caused by an existing operation of HELICS that was not terminated. This was resolved by:

    - ps aux | grep "helic"
    - Then kill that process:
        - ps kill [PID]
