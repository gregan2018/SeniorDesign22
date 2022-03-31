function imOut = HistMatch(imIn, refCdf)

% Inputs
% imIn: input image to be contrast enhanced
% refCdf: reference cumulative distribution function to be applied

% Compute the counts for each pixel value in the image
counts = zeros(1, 256);
for idx = 1:numel(imIn)
    pixVal = imIn(idx) + 1;
    counts(pixVal) = counts(pixVal) + 1;
end

% Sum the counts and divide to find probaility of each pixel value
total = sum(counts);
prob = counts/total;

% Compute the cdf of the image and initalize the look up table
cdf = cumsum(prob);
lut = zeros(length(cdf), 1);

% Compare the reference cdf and image cdf to generate lookup table
for pixValIdx = 1:length(counts)
    idx = find(refCdf >= cdf(pixValIdx), 1, 'first');
    if ~isempty(idx)
        lut(pixValIdx) = idx - 1;
    end
end
 
%  Extrapolate endpoint value
if lut(end) == 0
    idx = find(lut, 1, 'last');
    lut(idx+1:end) = lut(idx);
end

% Match pixel values using the lookup table
imOut = imIn;
for idx = 1:numel(imOut)
    pixVal = imOut(idx) + 1;
    imOut(idx) = lut(pixVal);
end

return

