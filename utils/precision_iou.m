function precisions = precision_plot(positions, ground_truth, show)
	
	max_iou = 0.9;  %used for graphs in the paper
	min_iou = 0.1;
	interval = 0.02;
	precisions = zeros((max_iou-min_iou)/interval+1, 1);
	
	%calculate iuo
    iou = zeros(length(positions),1);
    for i=1:length(iou)
        iou(i) = overlap_ratio(positions(i,:), ground_truth(i,:));
    end

	%compute precisions
    p = min_iou;
	for  i= 1:length(precisions)
		precisions(i) = double(nnz(iou >= p)) / length(iou);
        p = p+interval;
	end
	
	%plot the precisions
	if show == 1,
% 		figure('Number','off', 'Name',['Precisions - ' title])
        figure
		plot([min_iou:interval:max_iou], precisions, 'r-', 'LineWidth',2)
        title('OTB100-Bolt')
		xlabel('IOU'), ylabel('Precision')
	end
	
end

