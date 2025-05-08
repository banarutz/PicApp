//
//  DenoisingView.swift
//  PicApp
//
//  Created by Sebastian Banaru on 07.01.2025.
//

import SwiftUI
import CoreML
import UIKit
import PhotosUI
import CoreImage
import CoreImage.CIFilterBuiltins

struct DenoisingView: View {
    @State private var inputImage: UIImage?
    @State private var denoisedImage: UIImage?
    @State private var isProcessing = false
    @State private var pickerDelegate: PickerDelegate?
    @State private var selectedModel: String? = "DnCNN"
    @State private var selectedClassicalMethod: String?
    @State private var processingTime: String = ""
    @State private var startTime: Date?
    @State private var processingProgress: Double = 0.0
    @State private var totalPatches: Int = 0
    @State private var processedPatches: Int = 0
    
    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading) {
                Text("Pick a model:")
                    .font(.headline)
                    .padding(.horizontal)
                    .padding(.top)
                
                HStack {
                    Button(action: {
                        selectedModel = "DnCNN"
                        selectedClassicalMethod = nil
                    }) {
                        Text("DnCNN")
                            .font(.system(size: 12))
                            .padding(.horizontal)
                            .padding(.vertical, 8)
                            .background(selectedModel == "DnCNN" ? Color.blue : Color.gray.opacity(0.3))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                    Button(action: {
                        selectedModel = "DnCNNSep"
                        selectedClassicalMethod = nil
                    }) {
                        Text("DnCNNSep")
                            .font(.system(size: 12))
                            .padding(.horizontal)
                            .padding(.vertical, 8)
                            .background(selectedModel == "DnCNNSep" ? Color.blue : Color.gray.opacity(0.3))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                }
                .padding(.horizontal)
                .padding(.bottom)
                
                Text("Classical Methods:")
                    .font(.headline)
                    .padding(.horizontal)
                
                ScrollView(.horizontal) {
                    HStack {
                        Button(action: {
                            selectedClassicalMethod = "blur"
                            selectedModel = nil
                        }) {
                            Text("Blur (Gaussian)")
                                .font(.system(size: 12))
                                .padding(.horizontal)
                                .padding(.vertical, 8)
                                .background(selectedClassicalMethod == "blur" ? Color.blue : Color.gray.opacity(0.3))
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                        Button(action: {
                            selectedClassicalMethod = "median"
                            selectedModel = nil
                        }) {
                            Text("Median Filter")
                                .font(.system(size: 12))
                                .padding(.horizontal)
                                .padding(.vertical, 8)
                                .background(selectedClassicalMethod == "median" ? Color.blue : Color.gray.opacity(0.3))
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                        Button(action: {
                            selectedClassicalMethod = "bilateral"
                            selectedModel = nil
                        }) {
                            Text("Bilateral Filter")
                                .font(.system(size: 12))
                                .padding(.horizontal)
                                .padding(.vertical, 8)
                                .background(selectedClassicalMethod == "bilateral" ? Color.blue : Color.gray.opacity(0.3))
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.bottom)
                }
            }
            .background(Color(.systemGray6))
            .zIndex(1)
            
            ScrollView {
                VStack(spacing: 20) {
                    if isProcessing {
                        ProgressView(value: processingProgress) {
                            Text("Processing...")
                        }
                        .scaleEffect(2)
                    } else if let denoised = denoisedImage {
                        VStack(spacing: 10) {
                            Image(uiImage: denoised)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(height: 300)
                                .overlay(Text("Denoised Image").foregroundColor(.white).background(Color.black.opacity(0.6)), alignment: .top)
                            Text(processingTime)
                                .font(.caption)
                                .foregroundColor(.gray)
                            Button("Load image") {
                                resetState(keepSelection: true)
                                showImagePicker()
                            }
                            .padding()
                            .buttonStyle(.borderedProminent)
                        }
                    } else if let input = inputImage {
                        VStack(spacing: 10) {
                            Image(uiImage: input)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(height: 300)
                                .overlay(Text("Original Image").foregroundColor(.white).background(Color.black.opacity(0.6)), alignment: .top)
                            
                            if selectedModel != nil || selectedClassicalMethod != nil {
                                Button("Run Denoising") {
                                    startTime = Date()
                                    runDenoising()
                                }
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                            }
                        }
                    } else {
                        Button("Load image") {
                            resetState()
                            showImagePicker()
                        }
                        .padding()
                        .buttonStyle(.borderedProminent)
                    }
                    Spacer()
                }
                .padding()
            }
        }
        .frame(maxHeight: .infinity, alignment: .top)
        .onChange(of: selectedModel) { newValue in
            if denoisedImage != nil && newValue != nil {
                denoisedImage = nil
            }
        }
        .onChange(of: selectedClassicalMethod) { newValue in
            if denoisedImage != nil && newValue != nil {
                denoisedImage = nil
            }
        }
    }
    
    func resetState(keepSelection: Bool = false) {
        inputImage = nil
        denoisedImage = nil
        processingTime = ""
        processingProgress = 0.0
        totalPatches = 0
        processedPatches = 0
        if !keepSelection {
            selectedModel = nil
            selectedClassicalMethod = nil
        }
    }
    
    func showImagePicker() {
        let config = PHPickerConfiguration()
        let picker = PHPickerViewController(configuration: config)
        let delegate = PickerDelegate { image in
            inputImage = image
            denoisedImage = nil
            processingTime = ""
            processingProgress = 0.0
            totalPatches = 0
            processedPatches = 0
        }
        picker.delegate = delegate
        pickerDelegate = delegate
        
        if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let root = scene.windows.first?.rootViewController {
            root.present(picker, animated: true)
        }
    }
    
    func runDenoising() {
        guard let input = inputImage else { return }
        isProcessing = true
        denoisedImage = nil
        processingTime = ""
        startTime = Date()
        processingProgress = 0.0
        processedPatches = 0
        
        let patchSize = 50
        if let cgImage = input.cgImage {
            let width = cgImage.width
            let height = cgImage.height
            totalPatches = (width + patchSize - 1) / patchSize * ((height + patchSize - 1) / patchSize)
            processingProgress = 0.0
            
            if let modelName = selectedModel {
                denoiseWithDnCNN(image: input, modelName: modelName) { processedImage, duration in
                    updateUI(with: processedImage, duration: duration)
                    isProcessing = false
                }
            } else if let method = selectedClassicalMethod {
                applyClassicalFilter(image: input) { processedImage, duration in
                    updateUI(with: processedImage, duration: duration)
                    isProcessing = false
                }
            }
        } else {
            isProcessing = false
        }
    }
    
    func updateUI(with image: UIImage?, duration: TimeInterval) {
        DispatchQueue.main.async {
            denoisedImage = image
            isProcessing = false
            processingProgress = 0.0
            totalPatches = 0
            processedPatches = 0
            processingTime = String(format: "Elapsed time: %.2f seconds", duration)
        }
    }
    
    func denoiseWithDnCNN(image: UIImage, modelName: String, completion: @escaping (UIImage?, TimeInterval) -> Void) {
            guard let cgImage = image.cgImage else {
                completion(nil, 0)
                return
            }

            let width = cgImage.width
            let height = cgImage.height
            let patchSize = 50
            let totalPatchesCount = ((height + patchSize - 1) / patchSize) * ((width + patchSize - 1) / patchSize)
            var patches: [(x: Int, y: Int, patch: MLMultiArray?)] = Array(repeating: (0, 0, nil), count: totalPatchesCount)
            let queue = DispatchQueue(label: "patchExtraction", attributes: .concurrent)

            let startPatchExtraction = Date()
            DispatchQueue.concurrentPerform(iterations: (height + patchSize - 1) / patchSize) { row in
                for col in 0..<(width + patchSize - 1) / patchSize {
                    let x = col * patchSize
                    let y = row * patchSize
                    if let mlArray = extractPatch(from: image, at: x, y: y, size: patchSize) {
                        let index = row * ((width + patchSize - 1) / patchSize) + col
                        queue.async(flags: .barrier) {
                            patches[index] = (x, y, mlArray)
                        }
                    } else {
                        print("Eroare la extragerea patch-ului la (\(x), \(y))")
                    }
                }
            }

            queue.sync(flags: .barrier) {}

            let endPatchExtraction = Date()
            let patchExtractionTime = endPatchExtraction.timeIntervalSince(startPatchExtraction)
            print("Timp extragere patch-uri (paralel): \(patchExtractionTime)")
            print("Running model: \(modelName)...")
            let modelFileName: String
            switch modelName {
            case "DnCNN":
                modelFileName = "dncnn-epoch=028-val_loss=0.0001"
            case "DnCNNSep":
                modelFileName = "dncnnsep_e49"
            default:
                print("Model necunoscut: \(modelName)")
                completion(nil, 0)
                return
            }

            guard let modelURL = Bundle.main.url(forResource: modelFileName, withExtension: "mlmodelc"),
                  let coremlModel = try? MLModel(contentsOf: modelURL) else {
                print("Model not found: \(modelFileName).mlmodelc")
                completion(nil, 0)
                return
            }

            let validPatches = patches.compactMap { $0.2 != nil ? ($0.0, $0.1, $0.2!) : nil }

            let startInference = Date()
            let outImage = reconstructImage(from: validPatches, model: coremlModel, modelName: modelName, originalSize: CGSize(width: width, height: height), patchSize: patchSize) { processedCount in
                DispatchQueue.main.async {
                    processedPatches = processedCount
                    if totalPatches > 0 {
                        processingProgress = Double(processedPatches) / Double(totalPatches)
                    }
                }
            }
            let endInference = Date()
            let inferenceTime = endInference.timeIntervalSince(startInference)
            print("Timp inferență: \(inferenceTime)")

            completion(outImage, patchExtractionTime + inferenceTime)
        }
    
    func applyClassicalFilter(image: UIImage, completion: @escaping (UIImage?, TimeInterval) -> Void) {
        guard let ciImage = CIImage(image: image) else {
            completion(nil, 0)
            return
        }
        
        let startTime = Date()
        var outputCIImage: CIImage?
        
        switch selectedClassicalMethod {
            case "blur":
                outputCIImage = ciImage.applyingGaussianBlur(sigma: 5)
            case "median":
                if let medianFilter = CIFilter(name: "CIMedianFilter") {
                    medianFilter.setValue(ciImage, forKey: kCIInputImageKey)
                    outputCIImage = medianFilter.outputImage
                }
        case "bilateral":
            let blurred = ciImage.applyingGaussianBlur(sigma: 3)
            if let sharpenFilter = CIFilter(name: "CIUnsharpMask") {
                sharpenFilter.setValue(blurred, forKey: kCIInputImageKey)
                sharpenFilter.setValue(2.5, forKey: kCIInputIntensityKey)
                sharpenFilter.setValue(1.0, forKey: kCIInputRadiusKey)
                outputCIImage = sharpenFilter.outputImage
            } else {
                    print("CIBilateralBlur filter is not available.")
                }
            default:
                outputCIImage = ciImage
            }
        
        var outputUIImage: UIImage?
        if let output = outputCIImage {
            let context = CIContext()
            if let cgImage = context.createCGImage(output, from: output.extent) {
                outputUIImage = UIImage(cgImage: cgImage)
            }
        }
        let endTime = Date()
        let duration = endTime.timeIntervalSince(startTime)
        completion(outputUIImage, duration)
    }
    
    func extractPatch(from image: UIImage, at x: Int, y: Int, size: Int) -> MLMultiArray? {
        guard let cgImage = image.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height
        
        let cropRect = CGRect(x: x, y: y, width: min(size, width - x), height: min(size, height - y))
        guard let croppedCG = cgImage.cropping(to: cropRect) else { return nil }
        
        UIGraphicsBeginImageContextWithOptions(CGSize(width: size, height: size), false, 1.0)
        let context = UIGraphicsGetCurrentContext()!
        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))
        UIImage(cgImage: croppedCG).draw(in: CGRect(origin: .zero, size: CGSize(width: cropRect.width, height: cropRect.height)))
        let paddedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        guard let inputArray = paddedImage?.toMLMultiArray() else { return nil }
        
        inputArray.normalizeChannels(mean: [0.0, 0.0, 0.0], std: [1.0, 1.0, 1.0])
        
        return inputArray
    }
    
    
    func reconstructImage(from patches: [(x: Int, y: Int, patch: MLMultiArray)], model: MLModel, modelName: String, originalSize: CGSize, patchSize: Int, progressCallback: @escaping (Int) -> Void) -> UIImage? {
        let width = Int(originalSize.width)
        let height = Int(originalSize.height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        // Stocăm rezultatele procesate în paralel
        var patchResults = Array<(x: Int, y: Int, image: CGImage?)>(repeating: (0, 0, nil), count: patches.count)
        let queue = DispatchQueue(label: "progressCallback", attributes: .concurrent)
        
        DispatchQueue.concurrentPerform(iterations: patches.count) { index in
            let (x, y, inputArray) = patches[index]
            var outputArray: MLMultiArray? // Declaram outputArray aici
            
            if modelName == "DnCNN" {
                guard let result = try? model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input": inputArray])).featureValue(for: "var_252")?.multiArrayValue else {
                    return
                }
                outputArray = result
            }
            else if modelName == "DnCNNSep"{
                guard let result = try? model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input": inputArray])).featureValue(for: "var_374")?.multiArrayValue else {
                    return
                }
                outputArray = result
            }
            
            // Adăugăm printarea valorilor output pentru debugging, doar dacă outputArray nu este nil
            if let output = outputArray {
                
                for i in 0..<output.count {
                    inputArray[i] = output[i]
                }
                
                if let buffer = output.toCVPixelBuffer(width: patchSize, height: patchSize),
                   let img = UIImage(pixelBuffer: buffer),
                   let patchCG = img.cgImage {
                    patchResults[index] = (x, y, patchCG)
                }
            } else {
                print("Eroare la obținerea output-ului pentru patch-ul (\(x), \(y)) - Model: \(modelName)")
            }
            
            queue.async(flags: .barrier) {
                progressCallback(index + 1)
            }
        }
        
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            return nil
        }
        
        for result in patchResults {
            if let patchCG = result.image {
                context.draw(patchCG, in: CGRect(x: result.x, y: result.y, width: patchSize, height: patchSize))
            }
        }
        
        guard let finalImage = context.makeImage() else { return nil }
        return UIImage(cgImage: finalImage, scale: 1.0, orientation: .downMirrored)
    }
}

class PickerDelegate: NSObject, PHPickerViewControllerDelegate {
    let completion: (UIImage?) -> Void
    init(completion: @escaping (UIImage?) -> Void) {
        self.completion = completion
    }
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        guard let provider = results.first?.itemProvider, provider.canLoadObject(ofClass: UIImage.self) else {
            completion(nil)
            return
        }
        provider.loadObject(ofClass: UIImage.self) { object, _ in
            DispatchQueue.main.async {
                self.completion(object as? UIImage)
            }
        }
            }
        }

        extension UIImage {
            func toCVPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
                let attrs = [
                    kCVPixelBufferCGImageCompatibilityKey: true,
                    kCVPixelBufferCGBitmapContextCompatibilityKey: true
                ] as CFDictionary

                var pixelBuffer: CVPixelBuffer?
                let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                                kCVPixelFormatType_32ARGB, attrs,
                                                &pixelBuffer)
                guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
                    return nil
                }

                CVPixelBufferLockBaseAddress(buffer, [])
                defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

                guard let context = CGContext(
                    data: CVPixelBufferGetBaseAddress(buffer),
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                ) else {
                    return nil
                }

                UIGraphicsPushContext(context)
                self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
                UIGraphicsPopContext()

                return buffer
            }
        }

        extension MLMultiArray {
            func toCVPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
                var pixelBuffer: CVPixelBuffer?
                let attrs = [
                    kCVPixelBufferCGImageCompatibilityKey: true,
                    kCVPixelBufferCGBitmapContextCompatibilityKey: true
                ] as CFDictionary

                let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                                kCVPixelFormatType_32BGRA, attrs,
                                                &pixelBuffer)
                guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

                CVPixelBufferLockBaseAddress(buffer, [])
                guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
                    CVPixelBufferUnlockBaseAddress(buffer, [])
                    return nil
                }
                let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
                for y in 0..<height {
                    for x in 0..<width {
                        let r = UInt8(max(0, min(1, self[[0, 0, y as NSNumber, x as NSNumber]].floatValue)) * 255.0)
                        let g = UInt8(max(0, min(1, self[[0, 1, y as NSNumber, x as NSNumber]].floatValue)) * 255.0)
                        let b = UInt8(max(0, min(1, self[[0, 2, y as NSNumber, x as NSNumber]].floatValue)) * 255.0)

                        let pixelOffset = y * bytesPerRow + x * 4
                        let pixelPointer = baseAddress.advanced(by: pixelOffset)

                        pixelPointer.storeBytes(of: b, as: UInt8.self)
                        pixelPointer.advanced(by: 1).storeBytes(of: g, as: UInt8.self)
                        pixelPointer.advanced(by: 2).storeBytes(of: r, as: UInt8.self)
                        pixelPointer.advanced(by: 3).storeBytes(of: 255, as: UInt8.self) // alpha
                    }
                }

                CVPixelBufferUnlockBaseAddress(buffer, [])
                return buffer
            }

            func normalizeChannels(mean: [Float], std: [Float]) {
                assert(mean.count == 3 && std.count == 3, "mean and std must have 3 elements each")

                let height = self.shape[2].intValue
                let width = self.shape[3].intValue

                for c in 0..<3 {
                    for y in 0..<height {
                        for x in 0..<width {
                            let index: [NSNumber] = [0, NSNumber(value: c), NSNumber(value: y), NSNumber(value: x)]
                            let value = self[index].floatValue
                            let normalized = (value - mean[c]) / std[c]
                            self[index] = NSNumber(value: normalized)
                        }
                    }
                }
            }
        }


        extension UIImage {
            func toMLMultiArray() -> MLMultiArray? {
                guard let pixelBuffer = self.toCVPixelBuffer(width: 50, height: 50) else { return nil }

                let width = 50
                let height = 50
                guard let array = try? MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32) else { return nil }
                CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
                if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
                    let buffer = base.assumingMemoryBound(to: UInt8.self)
                    for y in 0..<height {
                        for x in 0..<width {
                            let offset = y * CVPixelBufferGetBytesPerRow(pixelBuffer) + x * 4
                            let r = Float(buffer[offset + 1]) / 255.0
                            let g = Float(buffer[offset + 2]) / 255.0
                            let b = Float(buffer[offset + 3]) / 255.0

                            array[[0, 0, y as NSNumber, x as NSNumber]] = NSNumber(value: r)
                            array[[0, 1, y as NSNumber, x as NSNumber]] = NSNumber(value: g)
                            array[[0, 2, y as NSNumber, x as NSNumber]] = NSNumber(value: b)
                        }
                    }
                }
                CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)

                return array
            }
        }

        extension UIImage {
            convenience init?(pixelBuffer: CVPixelBuffer) {
                let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
                let context = CIContext()
                if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                    self.init(cgImage: cgImage)
                } else {
                    return nil
                }
            }
        }
