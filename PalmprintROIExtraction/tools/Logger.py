import logging
import colorlog

log_colors_config = {
    'DEBUG': 'white',  # cyan white
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}


class Logger:
    def __init__(self, save_path=None):
        logger = logging.getLogger('logger_name')

        # 输出到文件
        if save_path is not None:
            file_handler = logging.FileHandler(filename=save_path, mode='a', encoding='utf8')
            file_handler.setLevel(logging.INFO)
            # 日志输出格式
            file_formatter = logging.Formatter(
                fmt='[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
                datefmt='%Y-%m-%d  %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            if not logger.handlers:
                logger.addHandler(file_handler)
            file_handler.close()

        # 输出到控制台
        console_handler = logging.StreamHandler()
        # 日志级别，logger 和 handler以最高级别为准，不同handler之间可以不一样，不相互影响
        logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
        console_formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
            datefmt='%Y-%m-%d  %H:%M:%S',
            log_colors=log_colors_config
        )
        console_handler.setFormatter(console_formatter)

        logger.addHandler(console_handler)
        self.logger = logger

    def get_logger(self):
        return self.logger


logger = Logger(save_path='test.log').get_logger()

if __name__ == '__main__':
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')
