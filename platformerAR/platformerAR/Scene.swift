//
//  Scene.swift
//  platformerAR
//
//  Created by Kevin Ciampaglia on 11/11/17.
//  Copyright © 2017 Cutie Hackers. All rights reserved.
//

import SpriteKit
import ARKit

class Scene: SKScene {
    
    override func didMove(to view: SKView) {
        // Setup your scene here
    }
    
    override func update(_ currentTime: TimeInterval) {
        // Called before each frame is rendered
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        var touchPlayer: Bool = false
        for touch in touches{
            let location = touch.location(in: self)
            for node in nodes(at: location){
                if(node is Player){
                    touchPlayer = true
                }
            }
        }
        if(touchPlayer == false){
            addAnchor()
        }
}

    func addAnchor(){
        guard let sceneView = self.view as? ARSKView else {
            return
        }
        
        // Create anchor using the camera's current position
        if let currentFrame = sceneView.session.currentFrame {
            var zDistance:Float = -0.2
            if( Helpers.setUpState == .addPlayer){
                zDistance = -1
            }
            
            // Create a transform with a translation of 0.2 meters in front of the camera
            var translation = matrix_identity_float4x4
            translation.columns.3.z = zDistance
            let transform = simd_mul(currentFrame.camera.transform, translation)
            
            // Add a new anchor to the session
            let anchor = ARAnchor(transform: transform)
            sceneView.session.add(anchor: anchor)
        }
    }

}
