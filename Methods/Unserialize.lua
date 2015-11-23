function unserialize(input)
	local topology = {}
	for i=1,#input do
		topology[i] = #input[i]
	end
	local tempNet = Net(topology)
	tempNet.unserialize(input)
	return tempNet
end