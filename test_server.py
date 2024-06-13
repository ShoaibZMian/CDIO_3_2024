import socket
import signal
import sys

def start_server(host, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)  # Listen for up to 5 queued connections
    print(f"Server started on {host}:{port}")

    def signal_handler(sig, frame):
        print("\nServer is shutting down...")
        server.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while True:
        print("Waiting for a connection...")
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")

        try:
            while True:
                # Send a "ready" message to the client after connection
                client_socket.send(b"ready")
                print("Sent 'ready' message to client.")

                # Receive command from client
                command = client_socket.recv(4096).decode()
                if command:
                    print(f"Received command: {command}")

                    # Process the command and send a response back
                    response = f"Command '{command}' received and processed."
                    client_socket.send(response.encode())
                else:
                    # If no command is received, break the loop
                    print("No command received. Closing connection.")
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            client_socket.close()
            print(f"Connection with {addr} closed.")

if __name__ == "__main__":
    host = "0.0.0.0"  # Listen on all available interfaces
    port = 8080       # Use the original port or change if still needed

    start_server(host, port)