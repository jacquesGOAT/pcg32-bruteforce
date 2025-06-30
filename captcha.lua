local function makegui(text)
	-- This script creates a GUI in the middle of the screen with custom text and a close button.
	-- It is parented to the LocalPlayer's Chat GUI.

	-- Define the text to display on the GUI
	local displayText = text

	-- Get the LocalPlayer and their Chat GUI
	local player = game.Players.LocalPlayer
	local chatGui = player:WaitForChild("PlayerGui"):WaitForChild("Chat")

	-- Create a new ScreenGui
	local screenGui = Instance.new("ScreenGui")
	screenGui.Name = "CustomMessageGui"
	screenGui.Parent = chatGui

	-- Create a Frame to hold the text and button
	local frame = Instance.new("Frame")
	frame.Size = UDim2.new(0, 300, 0, 150)
	frame.Position = UDim2.new(0.5, -150, 0.5, -75)
	frame.BackgroundColor3 = Color3.fromRGB(40, 40, 40)
	frame.BorderSizePixel = 0
	frame.Parent = screenGui

	-- Create a TextBox to display the message and allow selection
	local textBox = Instance.new("TextBox")
	textBox.Size = UDim2.new(1, 0, 0.7, 0)
	textBox.Position = UDim2.new(0, 0, 0, 0)
	textBox.BackgroundTransparency = 1
	textBox.Text = displayText
	textBox.TextColor3 = Color3.fromRGB(255, 255, 255)
	textBox.TextScaled = false
	textBox.TextSize = 16
	textBox.Font = Enum.Font.SourceSansBold
	textBox.TextXAlignment = Enum.TextXAlignment.Center
	textBox.TextYAlignment = Enum.TextYAlignment.Center
	textBox.ClearTextOnFocus = false
	textBox.Parent = frame

	-- Create a close button
	local closeButton = Instance.new("TextButton")
	closeButton.Size = UDim2.new(0.2, 0, 0.2, 0)
	closeButton.Position = UDim2.new(0.9, -30, 0.1, 0)
	closeButton.BackgroundColor3 = Color3.fromRGB(255, 70, 70)
	closeButton.Text = "X"
	closeButton.TextColor3 = Color3.fromRGB(255, 255, 255)
	closeButton.TextScaled = true
	closeButton.Font = Enum.Font.SourceSansBold
	closeButton.Parent = frame

	-- Add functionality to the close button
	closeButton.MouseButton1Click:Connect(function()
		screenGui:Destroy()
	end)

end
local captchaGui = game.Players.LocalPlayer.PlayerGui:FindFirstChild("Captcha");
if not captchaGui then 
	print("No Gui")
	return
end 

local viewport = captchaGui.MainFrame.Viewport
local one = viewport.Ambient:ToHSV()
local two = viewport.LightColor:ToHSV()

makegui(tostring(one) .. " " .. tostring(two))

-- x's are wildcards. Through testing, one can find that there are potentially unwanted state changes in between your sequences, which can be ignored as such.
--./pcg32r -l 8 -b 0.3987556993961334 0.9557129740715027 x x x x 0.7293798327445984 0.6796949505805969 x x x x