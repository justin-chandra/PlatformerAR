//
//  Helpers.swift
//  platformerAR
//
//  Created by Omar Peraza on 11/11/17.
//  Copyright © 2017 Cutie Hackers. All rights reserved.
//

import Foundation
import SpriteKit

enum SetUpState:Int{
    case addPlayer, none
    
}

class Helpers{
    static var someString:String = ""
    static var setUpState: SetUpState = .addPlayer
}
