local function Neuron(numOutputs,myIndex)
	--[[Private]]--
	local m_outputVal = 0
	local m_outputWeights = {}
	local m_myIndex = myIndex
	local m_gradient = 0
	local eta = 0.15
	local alpha = 0.5
	local function transferFunction(x)
		return 1 / (1 + math.exp(-x /0.5))
		--return math.tanh(x)
	end
	local function transferFunctionDerivative(x)
		return 1-x*x
	end
	local function sumDOW(nextLayer)
		sum = 0
		for n=1,#nextLayer do
			sum = sum + m_outputWeights[n].weight * nextLayer[n].getGradient()

		end
		return sum
	end

	--[[Public]]--
	local self = {}
	function self.setInfo(tabl)
		m_gradient = tabl.gra
		m_outputWeights = tabl.wei
		m_outputVal = tabl.val
	end
	function self.getInfo()
		return {
			gra = m_gradient,
			wei = m_outputWeights,
			val = m_outputVal,
		}
	end
	function self.getGradient()
		return m_gradient
	end
	function self.setOutputVal(val)
		m_outputVal = val
	end
	function self.getOutputVal()
		return m_outputVal
	end
	function self.getOutputWeights()
		return m_outputWeights
	end
	function self.feedForward(prevLayer)
		local sum = 0
		for n=0,#prevLayer do
			sum = sum + prevLayer[n].getOutputVal()*(prevLayer[n].getOutputWeights())[m_myIndex].weight
		end
		m_outputVal = transferFunction(sum)
	end
	function self.calcOutputGradients(targetVal)
		local delta = targetVal - m_outputVal
		m_gradient = delta * transferFunctionDerivative(m_outputVal)
	end
	function self.calcHiddenGradients(nextLayer)
		local dow = sumDOW(nextLayer)
		m_gradient = dow*transferFunctionDerivative(m_outputVal)
	end
	function self.changeConn(index,deltaWeight,weight)
		m_outputWeights[index] = {weight = weight, deltaWeight = deltaWeight}
	end
	function self.updateInputWeights(prevLayer)
		for n=0,#prevLayer do
			local neuron = prevLayer[n]
			local oldDeltaWeight = (neuron.getOutputWeights())[m_myIndex].deltaWeight
			local newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha*oldDeltaWeight
			local newWeight = neuron.getOutputWeights()[m_myIndex].weight + newDeltaWeight
			neuron.changeConn(m_myIndex,newDeltaWeight,newWeight)
		end
	end

	--[[Constructor]]--
	for c=1,numOutputs do
		m_outputWeights[c] = {
			weight = math.random(),
			deltaWeight = 0,
		}
	end
	return self
end