load('/data/yxi/Documents/tiny/data/widerface/wider_face_split/wider_face_val.mat');
basedir = '/data/yxi/Documents/SSH_NCCL/SSH/data/datasets/wider/WIDER_val/images/';
destdir = '/data/yxi/Documents/SSH_NCCL/SSH/output/ssh/wider_val/SSH/vis_gt/';
for i = 1 : size(event_list,1)
   for j = 1 : size(file_list{i},1)
        imgName = [basedir, event_list{i},'/',file_list{i}{j},'.jpg'];
        img = imread(imgName);
        bbox = face_bbx_list{i}{j};
        thr = 0;
        vis_gt(img, bbox, thr);
        
        fileDir = [destdir, event_list{i}];
        if ~exist(fileDir)
            mkdir(fileDir);
        end
        saveName = [fileDir, '/',file_list{i}{j},'.png']
        saveas(gcf,saveName);
   end
end

