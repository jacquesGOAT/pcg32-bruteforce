local dialogueRemote = nil
local function fireDialogue(count)	
	for i,v in pairs(game.ReplicatedStorage.Requests:GetChildren()) do 
		if v:IsA("RemoteEvent") and v:GetAttribute("Conner") == true then
			dialogueRemote = v
			print("Found existing dialogue remote:", v:GetFullName())
			break
		end
	end

	if not dialogueRemote then 
		for i,v in pairs(game.ReplicatedStorage.Requests:GetChildren()) do 
			if v:IsA("RemoteEvent") then
				local c
				c = v.OnClientEvent:Connect(function(data)
					if dialogueRemote then 
						c:Disconnect()
					end 

					if type(data) == "table" and (data.speaker or data.exit) then 
						c:Disconnect()
						dialogueRemote = v
						dialogueRemote:SetAttribute("Conner", true)

						print("Found dialogue remote:", v:GetFullName())
					end
				end)
			end
		end
	end

	repeat task.wait() until dialogueRemote

	--[[if not game:FindFirstChild("Conner") then 
		local f = Instance.new("Folder")
		f.Name = "Conner"
		f.Parent = game

		local cnt = 0
		local c
		c = remote.OnClientEvent:Connect(function(...)
			if not game:FindFirstChild("Conner") then 
				c:Disconnect()
			else
				cnt = cnt + 1
				print(...)
				print("Called", cnt, "times")
			end
		end)
	end

	print("Grabbed remote")]]
	--local gateCount = 1 -- (to d4, back to skycastle 1)
	--local count = 22 - (8 + 2*gateCount) -- remove 8 (captcha) + 2 * gate count (assuming u do precise gates, since unprecise gates adds 1)
	for i = 1, count do 
		dialogueRemote:FireServer({choice="Can we talk about my feelings?"})

		task.wait(0.1);
	end
end