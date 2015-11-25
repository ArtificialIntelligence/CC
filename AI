--[[
  A new attempt at coding a neural Network
  Author: Wassil Janssen a.k.a. Creator
]]--
--[[Class Neuron]]
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

--[[Class Net]]
function Net(topology)
    --[[Private]]--
    local m_layers = {} --m_layers[layerNum][neuronNum]

    --[[Public]]--
    local self = {}
    function self.feedForward(inputVals)
      assert(#inputVals == #m_layers[1],"Mate, you are coding like a girl! Just joking, girls can code too... not like you!")
      for i=1,#inputVals do
        m_layers[1][i].setOutputVal(inputVals[i])
      end
      for i=2,#m_layers do
        for n=1,#m_layers[i] do
          m_layers[i][n].feedForward(m_layers[i-1])
        end
      end
    end
    function self.backProp(targetVals)
      --Calculate overall net error (RMS)
      local outputLayer = m_layers[#m_layers]
      local m_error = 0
      for n=1,#outputLayer do
        local delta = targetVals[n] - outputLayer[n].getOutputVal()
        m_error = m_error + delta*delta
      end
      m_error = m_error/#outputLayer
      m_error = math.sqrt(m_error)

      --Calculate output layer gradient
      for n=1,#outputLayer do
        outputLayer[n].calcOutputGradients(targetVals[n])
      end

      --Calculate gradients on hidden layers
      for layerNum = #m_layers-1,2,-1 do
        local hiddenLayer = m_layers[layerNum]
        local nextLayer = m_layers[layerNum+1]

        for n=0,#hiddenLayer do
          hiddenLayer[n].calcHiddenGradients(nextLayer)
        end
      end

      --Update connection weights
      for layerNum = #m_layers,2,-1 do
        local layer = m_layers[layerNum]
        local prevLayer = m_layers[layerNum - 1]

        for n=1,#layer do
          layer[n].updateInputWeights(prevLayer)
        end
      end

    end
    function self.getResults(resultVals)
      local resultVals = {}
      for n=1,#m_layers[#m_layers] do
        resultVals[#resultVals + 1] = m_layers[#m_layers][n].getOutputVal()
      end
      return resultVals
    end
    --[[Constructor]]--

    local numLayers = #topology
    for i=1,numLayers do
      local numOutputs = i == numLayers and 0 or topology[i+1]
      m_layers[i] = {}
      for n=1,topology[i] do
        m_layers[i][n] = Neuron(numOutputs,n)
        --print("Made a Neuron")
      end
      m_layers[i][0] = Neuron(numOutputs,0)
      m_layers[i][0].setOutputVal(1)
      --print("Made a Neuron")
      --print(numOutputs)
    end
    function self.serialize()
      local output = {}
      for i=1,#m_layers do
        output[i] = {}
        for m=1,#m_layers[i] do
          output[i][m] = m_layers[i][m].getInfo()
        end
      end
      return output
    end
    function self.unserialize(tabl)
      for i=1,#tabl do
        for m=1,#tabl[i] do
          m_layers[i][m].setInfo(tabl[i][m])
        end
      end
    end

    return self
  end

  function unserialize(input)
    local topology = {}
    for i=1,#input do
      topology[i] = #input[i]
    end
    local tempNet = Net(topology)
    tempNet.unserialize(input)
    return tempNet
  end