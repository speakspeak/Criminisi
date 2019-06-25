#import numpy as np
import struct
import logging

class image:
    def __init__(self):
        self.data_type = ''
        self.data = list()
        self.row = 0
        self.col = 0

    def set_image_size(self, row, col):
        self.row = row
        self.col = col

    def set_image_data_type(self, data_type):
        self.data_type = data_type

    def read_image_data(self,path):
        # clear list before reading.
        self.data.clear()
        # open the image file
        with open(path,'rb') as file_object:
            for i in range(self.col):
                #deal different data types.
                if self.data_type == "char":
                    #struct.unpack is used for converting binary to other type
                    self.data += list(struct.unpack('B'*self.row, file_object.read(self.row)))
                elif self.data_type == "int":
                    self.data += list(struct.unpack('I'*self.row, file_object.read(self.row * 4)))
                elif self.data_type == "float":
                    self.data += list(struct.unpack('f'*self.row, file_object.read(self.row * 4)))
                    
    def write_image_data(self,path):
        with open(path, 'wb') as file_object:
            if self.data_type == "char":
                for j in range(self.col):
                    for i in range(self.row):
                        tmp = struct.pack('B', self.data[j*self.row+i])
                        file_object.write(tmp)
                    #if j%100 == 0:
                    #    print(j," line ok!")

class deal:
    def __init__(self, image_rgb, image_micro, image_binary, image_confi, logger):
        self.image_rgb = image_rgb
        self.image_micro = image_micro
        self.image_binary = image_binary
        self.image_confi = image_confi
        self.row = image_binary.row
        self.col = image_binary.col
        self.logger = logger

    #the second point must be complete.
    def __calcu_ssd(self,x1, y1, x2, y2):
        ssd = 0
        if self.image_binary.data[y2*self.row+x2] != 0:
            return -1
        for i in range(-2,3):
            for j in range(-2,3):
                #if self.image_binary.data[(y2+i)*self.row+x2+j] != 0:
                    #if self.image_binary.data[(y1+i)*self.row+x1+j] != 0:
                #this need to write by numpy
                #for k in range(len(self.image_micro)):
                #    ssd += (self.image_micro[k].data[(y1+i)*self.row+x1+j] - self.image_micro[k].data[(y2+i)*self.row+x2+j]) * \
                #            (self.image_micro[k].data[(y1+i)*self.row+x1+j] - self.image_micro[k].data[(y2+i)*self.row+x2+j])
                if self.image_binary.data[(y1+i)*self.row+x1+j] == 0:
                    if self.image_binary.data[(y2+i)*self.row + x2 + j] == 0:
                        for k in range(len(self.image_rgb)):
                            ssd += (self.image_rgb[k].data[(y1+i)*self.row+x1+j] - self.image_rgb[k].data[(y2+i)*self.row+x2+j]) * \
                                   (self.image_rgb[k].data[(y1+i)*self.row+x1+j] - self.image_rgb[k].data[(y2+i)*self.row+x2+j])
                    else:
                        return -1
        return ssd

    def __calcu_confident(self, x, y):
        confident = 0.0
        for i in range(-2,3):
            for j in range(-2,3):
                confident = confident + self.image_confi.data[(j+y)*self.row+i+x]
        confident = confident * 1.0 / 25
        return confident

    def __calcu_data(self, x, y):
        data_x = 0.0
        data_y = 0.0
        sobel_x = [[-0.25,0.0,0.25],[-0.5,0.0,0.5],[-0.25,0.0,0.25]]
        sobel_y = [[0.25,0.5,0.25],[0.0,0.0,0.0],[-0.25,-0.5,-0.25]]
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(len(self.image_micro)):
                    data_x += sobel_x[i+1][j+1]*self.image_micro[k].data[(j+y)*self.row+i+x]
                    data_y += sobel_y[i+1][j+1]*self.image_micro[k].data[(j+y)*self.row+i+x]
        data = 0.01
        if data_x < 0 :
            data += -data_x
        else:
            data += data_x
        if data_y < 0:
            data += -data_y
        else:
            data += data_y
        return data
                    
    def __find_min(self,x,y):
        first_flag = False
        for i in range(2,self.row-2):
            for j in range(2,self.col-2):
                ssd_tmp = self.__calcu_ssd(x,y,i,j)
                if ssd_tmp != -1:
                    ssd_min_value = ssd_tmp
                    ssd_min_x = i
                    ssd_min_y = j
                    first_flag = True
                    break
            if first_flag == True:
                break

        for i in range(2,self.row-2):
            for j in range(2,self.col-2):
                ssd_tmp = self.__calcu_ssd(x,y,i,j)
                if ssd_tmp != -1:
                    if ssd_tmp < ssd_min_value:
                        ssd_min_value = ssd_tmp
                        ssd_min_x = i
                        ssd_min_y = j
                #else:
                    #print("ssd_tmp == -1",i,"   ",j)
        #print("x:",x," y:",y," OK!")
        #print("x:",ssd_min_x," y:",ssd_min_y)

        msg = "x: {0} y: {1} OK!".format(x,y)
        self.logger.info(msg)
        return ssd_min_x, ssd_min_y

    def final(self):
        #initial edge queue
        confident_edge_queue = list()
        for i in range(2, self.row-2):
            for j in range(2, self.col -2):
                if self.image_binary.data[j*self.row+i] != 0:
                    flag = 0
                    for n in range(-1,2):
                        for m in range(-1,2):
                            if self.image_binary.data[(j+m)*self.row+i+n] == 0:
                                confident = self.__calcu_confident(i,j)
                                data = self.__calcu_data(i,j)
                                priv = confident * data * 1.0
                                confi_struct = [priv,i,j,confident]
                                confident_edge_queue.append(confi_struct)
                                flag = 1
                                break
                        if flag == 1:
                            break
        confident_edge_queue.sort(reverse = True)
        #print("final first step done!")
        #print(len(confident_edge_queue))
        msg = "begin queue length {0}".format(len(confident_edge_queue))
        self.logger.info(msg)
        
        while len(confident_edge_queue) != 0:
            x = confident_edge_queue[0][1]
            y = confident_edge_queue[0][2]
            msg = "pop element x: {0}  y: {1}".format(x,y)
            self.logger.info(msg)
            
            ssd_min_x,ssd_min_y = self.__find_min(x,y)
            self.image_binary.data[y*self.row+x] = 0
            self.image_confi.data[y*self.row+x] = confident_edge_queue[0][3]
            self.image_rgb[0].data[y*self.row+x] = self.image_rgb[0].data[ssd_min_y*self.row + ssd_min_x]
            self.image_rgb[1].data[y*self.row+x] = self.image_rgb[1].data[ssd_min_y*self.row + ssd_min_x]
            self.image_rgb[2].data[y*self.row+x] = self.image_rgb[2].data[ssd_min_y*self.row + ssd_min_x]
            for i in range(-1,2):
                for j in range(-1,2):
                    if self.image_binary.data[(y+j)*self.row+x+i] != 0:
                        flag = 0
                        for m in range(len(confident_edge_queue)):
                            if ((y+j) == confident_edge_queue[m][2]) and ((x+i) == confident_edge_queue[m][1]):
                                flag = 1
                        if flag == 0:
                            confident = self.__calcu_confident(x+i,y+j)
                            data = self.__calcu_data(x+i,y+j)
                            priv = confident * data * 1.0 / 255
                            confi_struct = [priv,x+i,y+j,confident]
                            confident_edge_queue.append(confi_struct)
                            msg = "push element x: {0} y: {1}".format(x+i,y+j)
                            self.logger.info(msg)
                            
            confident_edge_queue.pop(0)
            msg = "new queue lengh {0}".format(len(confident_edge_queue))
            self.logger.info(msg)

            for m in range(len(confident_edge_queue)):
                cx = confident_edge_queue[m][1]
                cy = confident_edge_queue[m][2]
                cconfident = self.__calcu_confident(cx,cy)
                cdata = self.__calcu_data(cx,cy)
                cpriv = cconfident * cdata * 1.0 / 255
                confident_edge_queue[m][3] = cconfident
                confident_edge_queue[m][0] = cpriv
            confident_edge_queue.sort(reverse = True)
                
            
if __name__ == "__main__":
    #1365   1287

    #RGB image
    image_r = image()
    image_r.set_image_size(1365, 1287)
    image_r.set_image_data_type('char')
    image_r.read_image_data("LC8_clip_high_r.img")
    
    image_g = image()
    image_g.set_image_size(1365, 1287)
    image_g.set_image_data_type('char')
    image_g.read_image_data("LC8_clip_high_g.img")

    image_b = image()
    image_b.set_image_size(1365, 1287)
    image_b.set_image_data_type('char')
    image_b.read_image_data("LC8_clip_high_b.img")

    image_rgb = [image_r,image_g,image_b]

    #micro_wave image
    image_vv = image()
    image_vv.set_image_size(1365,1287)
    image_vv.set_image_data_type('char')
    image_vv.read_image_data("sentinel_VV_resample_char_high.img")

    image_vh = image()
    image_vh.set_image_size(1365,1287)
    image_vh.set_image_data_type('char')
    image_vh.read_image_data("sentinel_VH_resample_char_high.img")

    image_micro = [image_vh,image_vv]
    
    #Binary image
    image_binary = image()
    image_binary.set_image_size(1365, 1287)
    image_binary.set_image_data_type('char')
    image_binary.read_image_data("binary_high_try")
    #print(image_binary.data[765*1365+393])
    #image_binary.write_image_data("binary_tryc")

    image_confi = image()
    image_confi.set_image_size(1365, 1287);
    image_confi.set_image_data_type('float')
    for j in range(0,1287):
        for i in range(0,1365):
            if image_binary.data[j*image_binary.row+i] == 0:
                image_confi.data.append(1.0)
            else :
                image_confi.data.append(0.0)

    #print(image_confi.data[0])
    #print("read ok")

    #logging init
    logger = logging.getLogger('mylog')
    logger.setLevel(level = logging.DEBUG)
    handler = logging.FileHandler('log1.txt')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.addHandler(handler)

    logger.info("logger begin")
    d = deal(image_rgb, image_micro, image_binary, image_confi, logger)
    d.final()
    #write
    image_r.write_image_data("done_r")
    #print("first wirte ok!")
    image_g.write_image_data("done_g")
    #print("second wirte ok!")
    image_b.write_image_data("done_b")
