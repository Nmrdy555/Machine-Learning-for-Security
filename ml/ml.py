def generate_security_event(timestamp, event, user, ip_address):
    formatted_data = f"{timestamp},{event},{user},{ip_address}\n"
    return formatted_data


# Example usage:
timestamp = "2024-04-08 16:58:27"
event = "Successful login"
user = "user"
ip_address = "127.0.0.1"

formatted_data = generate_security_event(timestamp, event, user, ip_address)
print(formatted_data)
