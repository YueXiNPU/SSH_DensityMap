function vis_gt(img, bbox, thr)

if ~isempty(img), imshow(img,'border','tight','initialmagnification','fit'); end

hold on;
%set (gcf,'Position',[0,0,512,512])
axis normal;
for i = 1:size(bbox, 1)
  
  rectangle('position', bbox(i,:), ...
            'EdgeColor', 'r', 'LineWidth', 2);
end
hold off;
