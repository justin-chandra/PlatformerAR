//
//  Player.swift
//  platformerAR
//
//  Created by Omar Peraza on 11/11/17.
//  Copyright © 2017 Cutie Hackers. All rights reserved.
//

import Foundation
import SpriteKit


class Player: SKSpriteNode {
    
    var playerSprite:SKSpriteNode = SKSpriteNode()
    
    func setUp(with name:String){
        
        playerSprite = SKSpriteNode(imageNamed: name)
        
        let moveUp: SKAction = SKAction.move(by: CGVector(dx:0,dy:100), duration: 1)
        moveUp.timingMode = .easeOut
        let moveDown: SKAction = SKAction.move(by: CGVector(dx:0,dy:-100), duration: 1)
        moveDown.timingMode = .easeOut
        
        let seq:SKAction = SKAction.sequence([moveUp,moveUp]);
        let repeatForever: SKAction = SKAction.repeatForever(seq)
        playerSprite.run(repeatForever)
        self.addChild(playerSprite)
    }
}
