import socket


def send_command(command):
    target_host = "172.20.10.2" 
    target_port = 8080  

   
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(1)
    client.connect((target_host, target_port))
    # receive
    response = client.recv(4096)
    if response.decode() == "ready":
        print("Server is ready.")
    else:
        print("Server is not ready.")
    # send the command
    client.send(command.encode())
    print(f"Sent command: {command}")

if __name__ == "__main__":
    while True:
        cmd = input("Enter a command (e.g., 'forward10', 'backward20', or 'quit' to exit): ")
        if cmd.lower() == "quit":
            break
        send_command(cmd)
