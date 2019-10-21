//
//  File.swift
//  
//
//  Created by Vincent Esche on 10/21/19.
//

import Foundation

public struct MultiModal<Model, Value>
    where Model: Hashable
{
    let model: Model
    let value: Value
}
