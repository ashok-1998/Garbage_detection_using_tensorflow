import os

from twilio.rest import Client

# Your Account Sid and Auth Token from twilio.com/user/account
account_sid = 'ACdb6f8f3c71cac107615b6948f5ed7d53'
auth_token = 'a7b5ec953291c916b09dac73bb22ed96'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                	body='Object detection!',
                    from_='whatsapp:+14155238886',
                    to='whatsapp:+917022514654'
                 )

print(message.sid)