import paramiko
import os
import time

# 配置开发板连接信息
HOST = "192.168.10.2"  # 替换为实际IP
PORT = 22  # SSH默认端口
USER = "yan"  # 替换为实际用户名
PASSWORD = "147963"  # 替换为实际密码


def create_remote_directory(sftp, remote_dir):
    """递归创建远程目录"""
    try:
        sftp.stat(remote_dir)
        return  # 目录已存在
    except FileNotFoundError:
        pass

    # 递归创建父目录
    parent_dir = os.path.dirname(remote_dir)
    if parent_dir and parent_dir != '/':
        create_remote_directory(sftp, parent_dir)

    try:
        sftp.mkdir(remote_dir)
        print(f"创建远程目录: {remote_dir}")
    except Exception as e:
        print(f"创建目录失败: {remote_dir} - {str(e)}")
        raise


def send_file(local_path, remote_path):
    """发送文件到开发板，支持多次调用"""
    # 确保本地文件存在
    if not os.path.exists(local_path):
        print(f"本地文件不存在: {local_path}")
        return False

    transport = None
    sftp = None
    try:
        # 创建传输对象
        transport = paramiko.Transport((HOST, PORT))
        transport.connect(username=USER, password=PASSWORD)

        # 创建SFTP客户端
        sftp = paramiko.SFTPClient.from_transport(transport)

        # 确保远程目录存在（递归创建）
        remote_dir = os.path.dirname(remote_path)
        create_remote_directory(sftp, remote_dir)

        # 发送文件
        sftp.put(local_path, remote_path)
        print(f"文件已发送: {local_path} -> {remote_path}")
        execute_command(f"chmod +x {remote_path}")
        return True
    except Exception as e:
        print(f"发送文件失败: {str(e)}")
        return False
    finally:
        # 确保关闭连接
        if sftp:
            sftp.close()
        if transport:
            transport.close()
        # 添加短暂延迟
        time.sleep(0.5)


def execute_command(command):
    """执行命令并接收输出"""
    ssh = None
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(HOST, PORT, USER, PASSWORD)

        # 执行命令
        stdin, stdout, stderr = ssh.exec_command(command)

        # 获取输出
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')

        if error:
            print("错误信息:")
            print(error)
        return output.strip()
    except Exception as e:
        print(f"执行命令失败: {str(e)}")
        return ""
    finally:
        # 确保关闭连接
        if ssh:
            ssh.close()
        # 添加短暂延迟
        # time.sleep(0.5)


# 示例使用
if __name__ == "__main__":
    # # 1. 发送多个文件到开发板
    # files_to_send = [
    #     ("hello.txt", "/home/yan/workdir/receive/hello.txt"),
    #     ("generated_models/tvm/0c447931_device.so", "/home/yan/workdir/receive/0c447931_device.so"),
    #     ("cpp/validator-riscv", "/home/yan/workdir/receive/validator-riscv")
    # ]
    #
    # for local, remote in files_to_send:
    #     success = send_file(local, remote)
    #     if success:
    #         print(f"文件 {local} 发送成功")
    #     else:
    #         print(f"文件 {local} 发送失败")
    #     # 添加短暂延迟
    #     time.sleep(1)
    # check_permission = execute_command("ls -l /home/yan/workdir/receive/validator-riscv")
    # print(f"文件权限: {check_permission}")
    # 2. 执行命令并获取输出
    result = execute_command("ls")
    print("开发板输出:")
    print(result)