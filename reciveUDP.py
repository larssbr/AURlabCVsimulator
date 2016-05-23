

import socket
import ctypes

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    print "received message:", data


class reciveUDP:
    # Comunication methods

    def __init__(self, MESSAGE):
        #self.MESSAGE = ObstacleAvoidance.getMessage()
        #self.MESSAGE = MESSAGE
        self.dataMessage = None


    def reciveUDPmessage(self):
        # addressing information of target
        # Remember extreact this to GUI level of code
        UDP_IP = "127.0.0.1"
        UDP_PORT = 1130

        #print "message:", self.MESSAGE

        # initialize a socket, think of it as a cable
        # SOCK_DGRAM specifies that this is UDP
        try:
            #sock = socket.socket(socket.AF_INET,  # Internet
                                 #socket.SOCK_DGRAM)  # UDP
            # send the command
            #sock.sendto(self.MESSAGE, (UDP_IP, UDP_PORT))
            ctypes.windll.shell32.IsUserAnAdmin()
            sock = socket.socket(socket.AF_INET,  # Internet
                                 socket.SOCK_DGRAM)  # UDP
            #sock.bind((UDP_IP, UDP_PORT))

            while True:
                self.dataMessage, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
                print "received message:", data

            # close the socket
            #sock.close()
        except:
            pass

    def printUDPmessage(self):
        print self.dataMessage

def main():
    ctypes.windll.shell32.IsUserAnAdmin()

    reciveUDPclass = reciveUDP()
    reciveUDPclass.reciveUDPmessage()
    reciveUDPclass.printUDPmessage()

if __name__ == '__main__':
    main()