import blpapi
from time import sleep

def keep_bloomberg_alive():
    session = blpapi.Session()
    if not session.start():
        print("Failed to start Bloomberg session.")
        return
    if not session.openService("//blp/refdata"):
        print("Failed to open service.")
        return

    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.append("securities", "IBM US Equity")
    request.append("fields", "PX_LAST")

    while True:
        session.sendRequest(request)
        sleep(600)  # ping every 10 minutes

if __name__ == "__main__":
    keep_bloomberg_alive()