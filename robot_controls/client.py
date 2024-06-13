import socket

def send_command(command):
    target_host = "172.20.10.4"
    target_port = 8080

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(1)
    client.connect((target_host, target_port))
    try:
            # receive the initial "ready" response
            response = client.recv(4096)
            if response.decode() == "ready":
                print("Server is ready.")
            else:
                print("Server is not ready.")

            # send the command
            client.send(command.encode())
            print(f"Sent command: {command}")

            # Set a longer timeout or remove it for waiting on the response
            client.settimeout(None)  # Wait indefinitely for the response

            # receive the result from the server
            while True:
                try:
                    result = client.recv(4096)
                    if result:
                        print("Received response from server: ", result.decode())
                        break
                except socket.timeout:
                    print("Still waiting for the server response...")
                    continue  # Keep waiting

    except socket.timeout:
        print("Connection timed out.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    while True:
        cmd = input(
            "Enter a command (e.g., 'forward10', 'backward20', or 'quit' to exit): "
        )
        if cmd.lower() == "quit":
            break
        send_command(cmd)