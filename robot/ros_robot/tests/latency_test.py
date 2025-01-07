import roslibpy
import time


def robot_disable_service(client):    
    service = roslibpy.Service(client, '/yk_destroyer/robot_disable', 'std_srvs/Trigger')
    request = roslibpy.ServiceRequest()

    print('Calling service...')
    time_1 = time.time()
    result = service.call(request)
    time_2 = time.time()
    print('Service response: {}'.format(result['message']))
    
    time_elapsed = time_2 - time_1
    
    return time_elapsed


def main():
    client = roslibpy.Ros(host='172.26.179.142', port=9090)
    client.run()

    time_elapsed = robot_disable_service(client)
    print(f"Time elapsed: {time_elapsed}")
    
    client.terminate()
    
if __name__ == "__main__":
    main()