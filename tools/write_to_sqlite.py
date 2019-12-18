import os
import cv2
import argparse
import base64
import sqlite3

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, help='directory to read data', default='data/faces_emore/images')
    parser.add_argument('--save_dir', type=str, help='path to save TFRecord file dir', default='data/faces_emore')
    return parser.parse_args()


def write_to_db(id, base64_code, label, cursor):
    cursor.execute('INSERT INTO IMAGEDATA (ID,DATA,LABEL) VALUES (?,?,?)', [id, base64_code, label])


def progress(directory, save_path):
    database_path = os.path.join(save_path, "train.db")
    assert not os.path.exists(database_path)
    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    class_indices = dict(zip(classes, range(len(classes))))

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IMAGEDATA\n'
                   '           (ID INT PRIMARY KEY     NOT NULL,\n'
                   '            DATA      VARBINARY    NOT NULL,\n'
                   '            LABEL     INT    NOT NULL);')

    cursor.execute('CREATE TABLE INFODATA\n'
                   '           (TOP_NUM     INT    NOT NULL,\n'
                   '            CLASS_NUM     INT    NOT NULL);')


    print("Table created successfully")

    # write to class sum
    write_to_db(-1, base64_code=bytes(0), label=len(classes), cursor=cursor)

    idx = 0
    for home, dirs, files in os.walk(directory):
        for filename in files:
            image_path = os.path.join(home, filename)
            image = cv2.imread(image_path)
            _, img_buffer = cv2.imencode(".jpg", image)
            base64_code = base64.b64encode(img_buffer.tostring())
            label = class_indices[os.path.basename(home)]
            write_to_db(idx, base64_code, label, cursor)
            idx = idx+1
            if idx%100==0:
                conn.commit()
                print("class progress {}/{}".format(label, len(classes)))
    cursor.execute('INSERT INTO INFODATA (TOP_NUM,CLASS_NUM) VALUES (?,?)', [idx, len(classes)])
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    args = get_args()
    progress(args.images_dir, args.save_dir)




