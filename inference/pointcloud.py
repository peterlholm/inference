"Generate colored pointclouds from rgb , mask and depth"
from PIL import Image

def nngenerate_pointcloud(rgb_file, mask_file,mydepth,ply_file):
    " generate the pointcloud"
    rgb = Image.open(rgb_file)
    depth = mydepth #np.load(depth_file )
    mask = Image.open(mask_file).convert('I')
    points = []    
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):

            color =   rgb.getpixel((v,u))

            if (mask.getpixel((v,u))<55):
                # Z = depth.getpixel((u, v))
                Z = depth[u,v]
                if Z < 0:
                    Z = 0 
                else:
                    if Z> 80:
                        Z = 80
                if Z == 0: 
                    continue
                Y = .306 * (v-80) *  Z/80 #.306 = tan(FOV/2) = tan(34/2)
                X = .306 * (u-80) *  Z/80
                points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    file = open(ply_file,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()
    