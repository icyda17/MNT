import os
import logging

filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs/ram.log')
logging.basicConfig(filename=filename, 
                    level=logging.DEBUG,
                    format ='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')