#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __init__ import *

class MLDA_HoloLens_Coordinate():
    def hololens_coordinate_server(self, req):
        if req.status == "visualize":
            rospy.loginfo("[Service em_mlda/hololens_coordinate] Start em_mlda/hololens_coordinate")
            
            resp = mlda_hololens_coordinateResponse()
            
            Pdz = np.loadtxt(os.path.join(DATA_FOLDER + "/learn_result", "Pdz.txt"), delimiter=',')
            
            current_category_prob = Pdz[req.object_number]
            
            current_category_prob = 1.0 / current_category_prob
            current_category_prob = current_category_prob / np.sum(current_category_prob)
            
            for target_category in range(len(current_category_prob)):
                distance = current_category_prob[target_category]
                x = random.uniform(0, distance)
                
                if(target_category%2 == 1):
                    x = -x

                y = math.sqrt(abs(current_category_prob[target_category] - pow(x, 2)))

                if(target_category%2 == 0):
                    y = -y
                
                target_coordinate = Point(x,y,0.)
                resp.coordinate.append(target_coordinate)
            
            resp.finish = True
            
            return resp

        else:
            rospy.logwarn(
                "[Service em_mlda/hololens_coordinate] command not found: %s", req.status)

        return mlda_hololensResponse(True)
    
    def __init__(self):
        s = rospy.Service('em_mlda/hololens_coordinate', mlda_hololens_coordinate, self.hololens_coordinate_server)
        rospy.loginfo("[Service em_mlda/hololens_coordinate] Ready em_mlda/hololens_coordinate")

if __name__ == "__main__":
    rospy.init_node('em_mlda_hololens_coordinate_server')
    MLDA_HoloLens_Coordinate()
    rospy.spin()